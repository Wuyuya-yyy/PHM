"""Automatic Phase-1 report generation for PHM modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    """Generate a Markdown report in mathematical modeling and IEEE style."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the report generator.

        Args:
            project_root: PHM project root.
        """
        self.project_root = project_root
        self.report_path = project_root / "reports" / "phase1_report.md"

    def generate(
        self,
        data_summary: Dict[str, Any],
        eda_results: Dict[str, Any],
        hi_results: Dict[str, Any],
        stage_results: Dict[str, Any],
        model_results: Dict[str, Any],
        bearing_results: Dict[str, Any] | None = None,
        transfer_health_results: Dict[str, Any] | None = None,
        bearing_model_results: Dict[str, Any] | None = None,
        deep_transfer_results: Dict[str, Any] | None = None,
    ) -> Path:
        """Generate the Phase-1 analysis report.

        Args:
            data_summary: Data scanning summary.
            eda_results: EDA result dictionary.
            hi_results: HI construction results.
            stage_results: Stage segmentation results.
            model_results: Degradation modeling results.
            bearing_results: XJTU-SY bearing feature-engineering summary.

        Returns:
            Report file path.
        """
        lines: List[str] = []
        lines.append("# Phase-1 Report: Satellite Reaction Wheel PHM Based on Shared Degradation Representation")
        lines.append("")
        lines.append("## Abstract")
        lines.append(
            "This report establishes the first-stage engineering baseline for predictive health management "
            "of satellite reaction wheels. The workflow includes automatic data profiling, degradation-oriented "
            "exploratory analysis, interpretable health index construction, stage segmentation, and initial "
            "physical/data-driven degradation modeling."
        )
        lines.append("")
        lines.append("## 1. Data Description")
        lines.append(f"Source directory: `{data_summary.get('source_dir', '')}`")
        lines.append("")
        for file_info in data_summary.get("data_files", []):
            lines.append(
                f"- `{file_info.get('file_name')}`: shape={file_info.get('shape')}, "
                f"columns={file_info.get('columns', file_info.get('variables', []))}"
            )
        lines.append("")
        lines.append("## 2. Exploratory Degradation Analysis")
        for dataset, info in eda_results.items():
            lines.append(f"### {dataset}")
            stats = info.get("statistics", {})
            lines.append(f"Missing values: `{stats.get('missing_values', {})}`")
            lines.append(f"Z-score outlier counts: `{stats.get('outlier_counts_z3', {})}`")
            self._append_figures(lines, info.get("figures", [])[:8])
        lines.append("")
        lines.append("## 3. Health Index Construction")
        lines.append(
            "The initial health index HI(t) is constructed by direction-consistent Min-Max normalization "
            "and one-dimensional PCA fusion. The AutoEncoder interface is reserved for later "
            "cross-modal representation learning and is not part of the completed training pipeline."
        )
        for dataset, info in hi_results.items():
            meta = info.get("metadata", {})
            lines.append(f"### {dataset}")
            lines.append(f"Feature columns: `{meta.get('feature_columns')}`")
            lines.append(f"PCA explained variance ratio: `{meta.get('pca_explained_variance_ratio'):.4f}`")
            self._append_figures(lines, info.get("figures", []))
        lines.append("")
        lines.append("## 4. Degradation Stage Segmentation")
        lines.append(
            "Three segmentation strategies are compared: HI threshold segmentation, derivative-based "
            "segmentation, and a Gaussian-evidence Bayesian-style change point detector."
        )
        for dataset, info in stage_results.items():
            lines.append(f"### {dataset}")
            lines.append(f"Stage boundaries: `{info.get('boundaries')}`")
            self._append_figures(lines, info.get("figures", []))
        lines.append("")
        lines.append("## 5. Initial Degradation Models")
        lines.append("The physical model uses `i(t)=i0+a(exp(bt)-1)`. Data-driven baselines include Wiener drift and LSTM prediction.")
        for dataset, info in model_results.items():
            lines.append(f"### {dataset}")
            lines.append(f"Target variable: `{info.get('target')}`")
            for method in ["exponential", "wiener", "lstm_baseline"]:
                method_info = info.get(method, {})
                lines.append(f"- {method}: metrics=`{method_info.get('metrics')}`")
                figure = method_info.get("figure")
                if figure:
                    self._append_figures(lines, [figure])
        lines.append("")
        lines.append("## 6. Preliminary RUL Analysis")
        lines.append(
            "At this stage, RUL analysis is based on monotonic HI progression and degradation model extrapolation. "
            "A probabilistic RUL module is reserved under `rul_prediction/` for uncertainty-aware prediction after "
            "bearing data and cross-domain samples are added."
        )
        lines.append("")
        lines.append("## 7. Extension Interfaces")
        lines.append(
            "The project reserves modular interfaces for transfer learning, multimodal fusion, shared latent "
            "space learning, Transformer-based forecasting, attention mechanisms, and probabilistic warning. "
            "The enhanced pipeline also includes a trained deep domain-adaptation experiment for DANN, CORAL, "
            "MMD, and AutoEncoder-joint comparison."
        )
        lines.append("")
        if bearing_results:
            lines.append("## 8. XJTU-SY Bearing Feature Engineering")
            lines.append(f"Current status: `{bearing_results.get('status')}`")
            lines.append(f"Processed files: `{bearing_results.get('n_files', 0)}`")
            lines.append(f"Operating conditions: `{bearing_results.get('n_conditions', 0)}`")
            lines.append(f"Bearing runs: `{bearing_results.get('n_bearings', 0)}`")
            lines.append(f"Latent feature shape: `{bearing_results.get('latent_shape', [])}`")
            lines.append(f"Feature groups: `{bearing_results.get('feature_groups', [])}`")
            lines.append(f"Channels: `{bearing_results.get('channels', [])}`")
            lines.append(f"Feature table: `{bearing_results.get('feature_path', '')}`")
            lines.append(f"HI table: `{bearing_results.get('hi_path', '')}`")
            lines.append(f"Latent table: `{bearing_results.get('latent_path', '')}`")
            self._append_figures(lines, bearing_results.get("figures", [])[:6])
            lines.append(
                "The bearing module provides time-domain, frequency-domain, and time-frequency degradation "
                "features for the future Bearing Encoder and shared latent degradation space."
            )
            lines.append("")
        if bearing_model_results:
            lines.append("## 9. XJTU-SY Bearing Degradation Model Comparison")
            lines.append(f"Processed bearing runs: `{bearing_model_results.get('n_bearings')}`")
            lines.append(f"Model comparison table: `{bearing_model_results.get('model_comparison_path')}`")
            lines.append(f"Mean metrics: `{bearing_model_results.get('mean_metrics')}`")
            lines.append(f"Best model counts: `{bearing_model_results.get('best_model_counts')}`")
            self._append_figures(lines, bearing_model_results.get("figures", []))
            lines.append("")
        if transfer_health_results:
            lines.append("## 10. Transfer Learning and Health Management")
            transfer = transfer_health_results.get("bearing_transfer", {})
            recommended = transfer_health_results.get("recommended_rul", {})
            warning = transfer_health_results.get("warning", {})
            similarity = transfer_health_results.get("domain_similarity", {})
            observed_similarity = transfer_health_results.get("observed_similarity", {})
            interval = transfer.get("transferred_rul_interval", [])
            lines.append(f"Task-1 baseline RUL: `{transfer.get('baseline_rul_days'):.1f}` days")
            lines.append(f"Bearing-transfer calibrated RUL: `{transfer.get('transferred_rul_days'):.1f}` days")
            if recommended:
                lines.append(f"Recommended combined-transfer RUL: `{recommended.get('rul_days'):.1f}` days")
                rec_interval = recommended.get("rul_interval", [])
                if len(rec_interval) == 2:
                    lines.append(f"Recommended RUL interval: `[{rec_interval[0]:.1f}, {rec_interval[1]:.1f}]` days")
            elif len(interval) == 2:
                lines.append(f"Transfer RUL interval: `[{interval[0]:.1f}, {interval[1]:.1f}]` days")
            if observed_similarity:
                lines.append(f"Observed 0-1800d similarity: `{observed_similarity.get('cosine_similarity'):.4f}`")
            lines.append(f"Cross-domain trajectory similarity: `{similarity.get('cosine_similarity'):.4f}`")
            lines.append(f"Trend correlation: `{similarity.get('trend_correlation'):.4f}`")
            lines.append(f"Current stage: `{transfer_health_results.get('current_stage')}`")
            lines.append(f"Warning level: `{warning.get('level')}`")
            lines.append(f"Recommended action: `{warning.get('recommended_action')}`")
            lines.append(
                "Method note: this health-management recommendation is based on physical-consistency "
                "degradation severity calibration; trained deep-transfer outputs are reported separately "
                "as a conservative validation reference."
            )
            self._append_figures(lines, transfer_health_results.get("figures", []))
            lines.append("")
        if deep_transfer_results:
            lines.append("## 11. Trained Deep Domain Adaptation")
            best = deep_transfer_results.get("best_method", {})
            lines.append("Trained methods: `no_adaptation`, `CORAL`, `MMD`, `DANN`, and `AutoEncoder-joint`.")
            lines.append(
                "Validation uses the known Attachment-1 full-life trajectory: the first 70% is used for "
                "training and the last 30% is used for real RUL error testing."
            )
            lines.append(f"Best trained method: `{best.get('method')}`")
            lines.append(f"Best A1 held-out RUL MAE: `{best.get('test_mae_days'):.1f}` days")
            lines.append(f"Best A2 deep-transfer reference RUL: `{best.get('a2_predicted_rul_days'):.1f}` days")
            lines.append(f"Deep-transfer report: `{deep_transfer_results.get('outputs', {}).get('report', '')}`")
            self._append_figures(lines, deep_transfer_results.get("figures", []))
            lines.append("")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path

    def _append_figures(self, lines: List[str], figures: List[str]) -> None:
        """Append relative Markdown image links.

        Args:
            lines: Report line buffer.
            figures: Absolute or relative figure paths.
        """
        for figure in figures:
            path = Path(figure)
            try:
                rel = path.relative_to(self.project_root)
            except ValueError:
                rel = path
            title = path.stem.replace("_", " ").title()
            lines.append(f"![{title}](../{rel.as_posix()})")
            lines.append("")
