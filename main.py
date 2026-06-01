"""Main entry point for the satellite flywheel PHM Phase-1 project."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from bearing_degradation_modeling import run_bearing_degradation_modeling
from bearing.feature_engineering import process_xjtu_sy
from data_scanner import DataScanner
from degradation_models import DegradationModeler
from deep_domain_adaptation import run_deep_domain_adaptation
from eda import FlywheelEDA
from health_index import HealthIndexBuilder
from report_generator import ReportGenerator
from stage_segmentation import StageSegmenter
from transfer_health_management import generate_transfer_health_management
from utils.io_utils import ensure_directories, load_config, setup_logger, write_json


def main() -> None:
    """Execute the full Phase-1 PHM analysis pipeline."""
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "configs" / "default_config.yaml")
    source_dir = Path(config["paths"]["source_data_dir"])
    bearing_dir = Path(config["paths"].get("bearing_data_dir", project_root / "raw_data" / "XJTU-SY"))
    ensure_directories(project_root)
    logger = setup_logger(project_root / "logs" / "phase1_pipeline.log")
    seed = int(config["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    dpi = int(config["project"]["dpi"])

    logger.info("Starting PHM Phase-1 pipeline")
    scanner = DataScanner(source_dir=source_dir, project_root=project_root, logger=logger)
    data_summary = scanner.scan()
    datasets = scanner.load_tabular_files()
    if not datasets:
        raise RuntimeError(f"No tabular datasets found under {source_dir}")

    eda_results = FlywheelEDA(project_root, dpi, logger).run(datasets)
    hi_results = HealthIndexBuilder(project_root, dpi, logger).run(datasets)
    stage_results = StageSegmenter(project_root, dpi, logger).run(hi_results)
    model_results = DegradationModeler(project_root, dpi, logger).run(datasets, hi_results)
    bearing_results = process_xjtu_sy(
        bearing_dir,
        project_root / "results",
        fs=float(config["analysis"].get("bearing_sampling_frequency", 25600.0)),
    )
    logger.info("Bearing module completed with status: %s", bearing_results.get("status"))
    bearing_model_results = run_bearing_degradation_modeling(project_root, dpi=dpi)
    logger.info("Bearing degradation modeling completed with status: %s", bearing_model_results.get("status"))
    transfer_health_results = generate_transfer_health_management(project_root, dpi=dpi)
    logger.info("Transfer health-management module completed with status: %s", transfer_health_results.get("status"))
    deep_transfer_results = run_deep_domain_adaptation(project_root, dpi=dpi)
    logger.info("Deep domain-adaptation module completed with status: %s", deep_transfer_results.get("status"))

    all_results = {
        "data_summary": data_summary,
        "eda": eda_results,
        "health_index": hi_results,
        "stage_segmentation": stage_results,
        "degradation_models": model_results,
        "bearing": bearing_results,
        "bearing_degradation_models": bearing_model_results,
        "transfer_health_management": transfer_health_results,
        "deep_domain_adaptation": deep_transfer_results,
    }
    write_json(all_results, project_root / "results" / "phase1_results.json")
    report_path = ReportGenerator(project_root).generate(
        data_summary,
        eda_results,
        hi_results,
        stage_results,
        model_results,
        bearing_results,
        transfer_health_results,
        bearing_model_results,
        deep_transfer_results,
    )
    logger.info("Phase-1 pipeline completed. Report: %s", report_path)


if __name__ == "__main__":
    main()
