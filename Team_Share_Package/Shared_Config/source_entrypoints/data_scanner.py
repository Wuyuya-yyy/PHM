"""Automatic data discovery and profiling for PHM datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utils.io_utils import write_json


class DataScanner:
    """Scan raw files and generate structured metadata summaries."""

    def __init__(self, source_dir: Path, project_root: Path, logger: logging.Logger) -> None:
        """Initialize the scanner.

        Args:
            source_dir: Directory containing contest problem and attachments.
            project_root: PHM project root.
            logger: Project logger.
        """
        self.source_dir = source_dir
        self.project_root = project_root
        self.logger = logger
        self.supported = {".csv", ".xlsx", ".txt", ".mat", ".npy"}

    def scan(self) -> Dict[str, Any]:
        """Scan all supported files and write data_summary.json.

        Returns:
            Data summary dictionary.
        """
        files = [p for p in self.source_dir.rglob("*") if p.is_file()]
        pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
        data_files = [p for p in files if p.suffix.lower() in self.supported]
        summary: Dict[str, Any] = {
            "source_dir": str(self.source_dir),
            "problem_pdf": [str(p) for p in pdfs],
            "data_files": [],
        }
        for path in data_files:
            try:
                summary["data_files"].append(self.profile_file(path))
            except Exception as exc:
                self.logger.exception("Failed to profile %s", path)
                summary["data_files"].append(
                    {"file_name": path.name, "path": str(path), "error": str(exc)}
                )
        write_json(summary, self.project_root / "data" / "data_summary.json")
        write_json(summary, self.project_root / "results" / "data_summary.json")
        return summary

    def profile_file(self, path: Path) -> Dict[str, Any]:
        """Profile one supported data file.

        Args:
            path: Data file path.

        Returns:
            Metadata dictionary containing dimensions, columns, dtypes, and missing values.
        """
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            df = self._read_table(path)
            return self._profile_dataframe(path, df)
        if suffix == ".xlsx":
            df = pd.read_excel(path)
            return self._profile_dataframe(path, df)
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
            return {
                "file_name": path.name,
                "path": str(path),
                "extension": suffix,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "missing_values": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else None,
            }
        if suffix == ".mat":
            data = loadmat(path)
            variables = {
                key: list(value.shape)
                for key, value in data.items()
                if not key.startswith("__") and hasattr(value, "shape")
            }
            return {
                "file_name": path.name,
                "path": str(path),
                "extension": suffix,
                "variables": variables,
            }
        raise ValueError(f"Unsupported file type: {suffix}")

    def load_tabular_files(self) -> List[Dict[str, Any]]:
        """Load all CSV/TXT/XLSX files as pandas DataFrames.

        Returns:
            List of dictionaries with file metadata and DataFrame objects.
        """
        tabular: List[Dict[str, Any]] = []
        for path in self.source_dir.rglob("*"):
            if path.suffix.lower() not in {".csv", ".txt", ".xlsx"}:
                continue
            try:
                df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else self._read_table(path)
                tabular.append({"name": self._stable_dataset_name(path, len(tabular) + 1), "path": path, "data": df})
            except Exception:
                self.logger.exception("Failed to load tabular file %s", path)
        return tabular

    @staticmethod
    def _stable_dataset_name(path: Path, index: int) -> str:
        """Create an ASCII dataset name for reproducible outputs.

        Args:
            path: Source data file path.
            index: One-based dataset index.

        Returns:
            Stable ASCII dataset name.
        """
        lower_name = path.name.lower()
        if "3500" in lower_name:
            return "attachment1_reaction_wheel_3500d_data"
        if "1800" in lower_name:
            return "attachment2_reaction_wheel_1800d_data"
        return f"dataset_{index:02d}_{path.stem.encode('ascii', errors='ignore').decode('ascii') or 'tabular'}"

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        """Read a delimited text file with automatic separator detection.

        Args:
            path: CSV or TXT path.

        Returns:
            Loaded DataFrame.
        """
        return pd.read_csv(path, sep=None, engine="python")

    @staticmethod
    def _profile_dataframe(path: Path, df: pd.DataFrame) -> Dict[str, Any]:
        """Profile a pandas DataFrame.

        Args:
            path: Source file path.
            df: DataFrame to profile.

        Returns:
            File metadata dictionary.
        """
        return {
            "file_name": path.name,
            "path": str(path),
            "extension": path.suffix.lower(),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(map(str, df.columns)),
            "missing_values": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
            "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            "numeric_describe": df.describe(include=[np.number]).round(6).to_dict(),
        }
