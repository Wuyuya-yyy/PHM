"""Physical and data-driven degradation models for PHM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from utils.plot_utils import save_figure, set_ieee_style


class DegradationModeler:
    """Fit exponential, Wiener, and LSTM-like baseline degradation models."""

    def __init__(self, project_root: Path, dpi: int, logger: logging.Logger) -> None:
        """Initialize the modeler.

        Args:
            project_root: PHM project root.
            dpi: Figure resolution.
            logger: Project logger.
        """
        self.project_root = project_root
        self.dpi = dpi
        self.logger = logger
        self.fig_dir = project_root / "figures" / "models"
        set_ieee_style()

    def run(self, datasets: list[Dict[str, Any]], hi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run degradation models on all datasets.

        Args:
            datasets: Loaded tabular datasets.
            hi_results: Health-index outputs.

        Returns:
            Model result dictionary.
        """
        output: Dict[str, Any] = {}
        for item in datasets:
            name = self._slug(item["name"])
            df = item["data"].copy()
            target = "current" if "current" in df.columns else df.select_dtypes(include=[np.number]).columns[-1]
            time_col = "day" if "day" in df.columns else None
            x = df[time_col].to_numpy(dtype=float) if time_col else np.arange(len(df), dtype=float)
            y = df[target].astype(float).interpolate().bfill().ffill().to_numpy()
            exp_result = self.fit_exponential(x, y, name, target)
            wiener_result = self.fit_wiener(x, y, name, target)
            lstm_result = self.fit_lstm_baseline(x, y, name, target)
            result = {"target": target, "exponential": exp_result, "wiener": wiener_result, "lstm_baseline": lstm_result}
            out_path = self.project_root / "results" / f"{name}_model_metrics.json"
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2, ensure_ascii=False)
            output[name] = result | {"result_path": str(out_path)}
        return output

    @staticmethod
    def exponential_model(t: np.ndarray, i0: float, a: float, b: float) -> np.ndarray:
        """Exponential degradation model i(t)=i0+a(exp(bt)-1).

        Args:
            t: Time values.
            i0: Initial value.
            a: Degradation amplitude.
            b: Degradation rate.

        Returns:
            Predicted signal.
        """
        t_norm = (t - np.min(t)) / (np.ptp(t) + 1e-12)
        return i0 + a * (np.exp(b * t_norm) - 1.0)

    def fit_exponential(self, x: np.ndarray, y: np.ndarray, name: str, target: str) -> Dict[str, Any]:
        """Fit and plot the physical exponential degradation model.

        Args:
            x: Time array.
            y: Target degradation signal.
            name: Dataset slug.
            target: Target variable name.

        Returns:
            Parameter and metric dictionary.
        """
        try:
            popt, _ = curve_fit(
                self.exponential_model,
                x,
                y,
                p0=[float(y[0]), float((y[-1] - y[0]) or 0.01), 1.0],
                maxfev=20000,
            )
            pred = self.exponential_model(x, *popt)
        except Exception as exc:
            self.logger.warning("Exponential fitting failed for %s: %s", name, exc)
            popt = [float(y[0]), 0.0, 0.0]
            pred = np.repeat(np.mean(y), len(y))
        metrics = self._metrics(y, pred)
        residual = y - pred
        fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
        axes[0].plot(x, y, color="#4A5568", linewidth=1.0, label="Observed")
        axes[0].plot(x, pred, color="#D62728", linewidth=1.3, label="Exponential Fit")
        axes[0].set_title("Exponential Degradation Model Fitting")
        axes[0].set_ylabel(target)
        axes[0].legend()
        axes[1].plot(x, residual, color="#2B6CB0", linewidth=1.0)
        axes[1].axhline(0, color="#4A5568", linewidth=0.7)
        axes[1].set_title("Residual Analysis")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Residual")
        figure = save_figure(self.fig_dir / f"{name}_{target}_exponential_fit.png", self.dpi)
        return {"parameters": [float(v) for v in popt], "metrics": metrics, "figure": figure}

    def fit_wiener(self, x: np.ndarray, y: np.ndarray, name: str, target: str) -> Dict[str, Any]:
        """Fit a Wiener-process drift baseline.

        Args:
            x: Time array.
            y: Target degradation signal.
            name: Dataset slug.
            target: Target variable name.

        Returns:
            Drift, diffusion, metrics, and figure path.
        """
        increments = np.diff(y)
        drift = float(np.mean(increments)) if len(increments) else 0.0
        diffusion = float(np.std(increments, ddof=1)) if len(increments) > 1 else 0.0
        pred = y[0] + drift * np.arange(len(y))
        metrics = self._metrics(y, pred)
        plt.figure(figsize=(6.6, 3.2))
        plt.plot(x, y, color="#4A5568", linewidth=1.0, label="Observed")
        plt.plot(x, pred, color="#2F855A", linewidth=1.3, label="Wiener Drift Baseline")
        plt.fill_between(x, pred - 1.96 * diffusion, pred + 1.96 * diffusion, color="#9AE6B4", alpha=0.35, label="95% Band")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.title("Wiener Process Degradation Baseline")
        plt.legend()
        figure = save_figure(self.fig_dir / f"{name}_{target}_wiener_baseline.png", self.dpi)
        return {"drift": drift, "diffusion": diffusion, "metrics": metrics, "figure": figure}

    def fit_lstm_baseline(self, x: np.ndarray, y: np.ndarray, name: str, target: str) -> Dict[str, Any]:
        """Train an LSTM if PyTorch is available, otherwise use autoregressive fallback.

        Args:
            x: Time array.
            y: Target degradation signal.
            name: Dataset slug.
            target: Target variable name.

        Returns:
            Prediction metrics and figure path.
        """
        seq_len = min(12, max(3, len(y) // 8))
        if len(y) <= seq_len + 5:
            pred = np.repeat(np.mean(y), len(y))
            model_name = "mean_fallback"
        else:
            try:
                pred = self._torch_lstm_predict(y, seq_len)
                model_name = "torch_lstm"
            except Exception as exc:
                self.logger.warning("LSTM unavailable for %s, using AR fallback: %s", name, exc)
                pred = self._autoregressive_predict(y, seq_len)
                model_name = "autoregressive_fallback"
        metrics = self._metrics(y[~np.isnan(pred)], pred[~np.isnan(pred)])
        plt.figure(figsize=(6.6, 3.2))
        plt.plot(x, y, color="#4A5568", linewidth=1.0, label="Observed")
        plt.plot(x, pred, color="#805AD5", linewidth=1.2, label=model_name)
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.title("LSTM Baseline Prediction")
        plt.legend()
        figure = save_figure(self.fig_dir / f"{name}_{target}_lstm_baseline.png", self.dpi)
        return {"model": model_name, "sequence_length": int(seq_len), "metrics": metrics, "figure": figure}

    @staticmethod
    def _torch_lstm_predict(y: np.ndarray, seq_len: int) -> np.ndarray:
        """Train a compact PyTorch LSTM and return in-sample one-step predictions.

        Args:
            y: Target array.
            seq_len: Sequence length.

        Returns:
            Prediction array with NaN warm-up values.
        """
        import torch
        from torch import nn

        torch.manual_seed(42)
        scaler = MinMaxScaler()
        ys = scaler.fit_transform(y.reshape(-1, 1)).astype("float32").ravel()
        xs = []
        targets = []
        for idx in range(seq_len, len(ys)):
            xs.append(ys[idx - seq_len : idx])
            targets.append(ys[idx])
        x_tensor = torch.tensor(np.asarray(xs)[:, :, None])
        y_tensor = torch.tensor(np.asarray(targets)[:, None])

        class TinyLSTM(nn.Module):
            """Small one-step LSTM regressor."""

            def __init__(self) -> None:
                """Initialize network layers."""
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
                self.head = nn.Linear(16, 1)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                """Run a forward pass.

                Args:
                    values: Input sequence tensor.

                Returns:
                    One-step prediction.
                """
                out, _ = self.lstm(values)
                return self.head(out[:, -1, :])

        model = TinyLSTM()
        optim = torch.optim.Adam(model.parameters(), lr=0.02)
        loss_fn = nn.MSELoss()
        for _ in range(120):
            optim.zero_grad()
            loss = loss_fn(model(x_tensor), y_tensor)
            loss.backward()
            optim.step()
        with torch.no_grad():
            pred_scaled = model(x_tensor).numpy().ravel()
        pred = np.full(len(y), np.nan)
        pred[seq_len:] = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        pred[:seq_len] = y[:seq_len]
        return pred

    @staticmethod
    def _autoregressive_predict(y: np.ndarray, seq_len: int) -> np.ndarray:
        """Use rolling linear regression as a deterministic fallback.

        Args:
            y: Target array.
            seq_len: Sequence length.

        Returns:
            In-sample prediction array.
        """
        pred = np.full(len(y), np.nan)
        pred[:seq_len] = y[:seq_len]
        for idx in range(seq_len, len(y)):
            local_x = np.arange(seq_len)
            coef = np.polyfit(local_x, y[idx - seq_len : idx], deg=1)
            pred[idx] = np.polyval(coef, seq_len)
        return pred

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute RMSE, MAE, and R-squared.

        Args:
            y_true: Observed values.
            y_pred: Predicted values.

        Returns:
            Metric dictionary.
        """
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _slug(text: str) -> str:
        """Convert text into a filesystem-safe slug.

        Args:
            text: Raw text.

        Returns:
            Sanitized slug.
        """
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
