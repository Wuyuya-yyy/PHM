"""Cross-modal shared latent degradation prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from encoders import BearingEncoder, FlywheelEncoder
from utils.plot_utils import save_figure, set_ieee_style

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class LatentPrototypeConfig:
    """Configuration for the cross-modal latent prototype."""

    latent_dim: int = 8
    random_seed: int = 42
    dpi: int = 320
    dropout: float = 0.15
    n_clusters: int = 3


class CrossModalSharedLatentPrototype:
    """Build flywheel embeddings and reserve bearing alignment into a shared latent space."""

    def __init__(self, config: Optional[LatentPrototypeConfig] = None) -> None:
        self.config = config or LatentPrototypeConfig()
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        self.scaler = StandardScaler()
        self.flywheel_encoder: Optional[FlywheelEncoder] = None
        self.bearing_encoder: Optional[BearingEncoder] = None

    def prepare_flywheel_features(self, hi_csv_path: Path) -> pd.DataFrame:
        """Create model-ready flywheel features from processed HI files."""
        data = pd.read_csv(hi_csv_path)
        feature_cols = [c for c in data.columns if c.endswith("_norm")]
        for base_col in ("HI_raw", "HI_smooth", "HI_derivative"):
            if base_col in data.columns:
                feature_cols.append(base_col)

        for col in [c for c in data.columns if c.endswith("_norm")]:
            data[f"{col}_rolling_mean"] = data[col].rolling(window=5, min_periods=1).mean()
            data[f"{col}_rolling_std"] = data[col].rolling(window=5, min_periods=1).std().fillna(0.0)
            data[f"{col}_derivative"] = data[col].diff().fillna(0.0)
            feature_cols.extend([f"{col}_rolling_mean", f"{col}_rolling_std", f"{col}_derivative"])

        prepared = data[["day"] + feature_cols].copy()
        prepared = prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return prepared

    def encode_flywheel(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Encode flywheel features with the PyTorch prototype encoder."""
        values = feature_frame.drop(columns=["day"], errors="ignore").to_numpy(dtype=np.float32)
        scaled = self.scaler.fit_transform(values).astype(np.float32)
        self.flywheel_encoder = FlywheelEncoder(
            input_dim=scaled.shape[1],
            latent_dim=self.config.latent_dim,
            dropout=self.config.dropout,
        )
        self.flywheel_encoder.eval()
        with torch.no_grad():
            embedding = self.flywheel_encoder(torch.from_numpy(scaled)).numpy()
        return embedding

    def build_bearing_encoder_interface(self, input_dim: int = 32) -> BearingEncoder:
        """Instantiate the reserved bearing encoder interface."""
        self.bearing_encoder = BearingEncoder(input_dim=input_dim, latent_dim=self.config.latent_dim)
        return self.bearing_encoder

    def analyze_latent_space(
        self,
        embedding: np.ndarray,
        days: np.ndarray,
        dataset_name: str,
        figure_dir: Path,
    ) -> Dict[str, object]:
        """Run PCA, t-SNE, optional UMAP, and clustering analysis."""
        figure_dir.mkdir(parents=True, exist_ok=True)
        cluster_labels = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_seed,
            n_init=10,
        ).fit_predict(embedding)

        pca_projection = PCA(n_components=2, random_state=self.config.random_seed).fit_transform(embedding)
        tsne_perplexity = max(2, min(30, (len(embedding) - 1) // 3))
        tsne_projection = TSNE(
            n_components=2,
            random_state=self.config.random_seed,
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(embedding)

        projections: Dict[str, np.ndarray] = {"PCA": pca_projection, "t-SNE": tsne_projection}
        umap_status = "skipped: umap-learn is not installed"
        try:
            import umap  # type: ignore

            projections["UMAP"] = umap.UMAP(
                n_components=2,
                random_state=self.config.random_seed,
                n_neighbors=min(15, max(2, len(embedding) - 1)),
            ).fit_transform(embedding)
            umap_status = "generated"
        except Exception:
            pass

        figure_paths: List[str] = []
        for method, projection in projections.items():
            figure_paths.append(
                self._plot_projection(
                    projection=projection,
                    color=days,
                    title=f"{dataset_name} latent trajectory by {method}",
                    color_label="Operating day",
                    path=figure_dir / f"{dataset_name}_latent_trajectory_{method.lower().replace('-', '')}.png",
                    line=True,
                )
            )
            figure_paths.append(
                self._plot_projection(
                    projection=projection,
                    color=cluster_labels,
                    title=f"{dataset_name} latent clusters by {method}",
                    color_label="Cluster",
                    path=figure_dir / f"{dataset_name}_latent_cluster_{method.lower().replace('-', '')}.png",
                    line=False,
                    discrete=True,
                )
            )

        stage_proxy = pd.qcut(days, q=3, labels=["Healthy", "Slow", "Accelerated"])
        figure_paths.append(
            self._plot_stage_separation(
                projection=pca_projection,
                stage_labels=stage_proxy.astype(str).to_numpy(),
                title=f"{dataset_name} latent stage separation",
                path=figure_dir / f"{dataset_name}_stage_separation_pca.png",
            )
        )

        return {
            "dataset": dataset_name,
            "embedding_shape": list(embedding.shape),
            "cluster_labels": cluster_labels.tolist(),
            "umap_status": umap_status,
            "figures": figure_paths,
        }

    def _plot_projection(
        self,
        projection: np.ndarray,
        color: np.ndarray,
        title: str,
        color_label: str,
        path: Path,
        line: bool,
        discrete: bool = False,
    ) -> str:
        set_ieee_style()
        plt.figure(figsize=(5.4, 3.7))
        palette = "viridis" if not discrete else "Set2"
        scatter = plt.scatter(projection[:, 0], projection[:, 1], c=color, s=18, cmap=palette, edgecolor="none")
        if line:
            plt.plot(projection[:, 0], projection[:, 1], color="#444444", linewidth=0.8, alpha=0.55)
        plt.title(title)
        plt.xlabel("Latent axis 1")
        plt.ylabel("Latent axis 2")
        if not discrete:
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_label)
        else:
            handles, _ = scatter.legend_elements()
            plt.legend(handles=handles, title=color_label, frameon=True)
        return save_figure(path, dpi=self.config.dpi)

    def _plot_stage_separation(
        self,
        projection: np.ndarray,
        stage_labels: np.ndarray,
        title: str,
        path: Path,
    ) -> str:
        set_ieee_style()
        plt.figure(figsize=(5.4, 3.7))
        sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=stage_labels, s=24, edgecolor="none")
        plt.title(title)
        plt.xlabel("PCA latent axis 1")
        plt.ylabel("PCA latent axis 2")
        plt.legend(title="Stage", frameon=True)
        return save_figure(path, dpi=self.config.dpi)
