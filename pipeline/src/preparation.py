from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import warnings
import pandas as pd
import numpy as np

import scipy.sparse as sp
import scanpy as sc
import squidpy as sq
import anndata as ad
import matplotlib.pyplot as plt
import anndata as ad

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import configure_logging

import gc

LOGGER = logging.getLogger(__name__)


def format_pipeline_step01_summary(*, 
                                   dataset_dir: str,
                                   metadata_dir: str,
                                   output_dir: str
                                   ) -> None:
    LOGGER.info(
            f"Pipeline Step01 summary:\n",
            f"- Original Dataset Directory: {dataset_dir}\n",
            f"- Metadata Directory: {metadata_dir}\n",
            f"- Output Directory: {output_dir}"
            )


def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step01 Data Preparation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    raw_dir = default_repo_root / "pipeline" / "data" / "raw"
    parser.add_argument("--metadata-dir", default=str(raw_dir / "metadata/"))
    parser.add_argument("--dataset-dir", default=str(raw_dir / "h5ad/"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step01(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    format_pipeline_step01_summary(
        dataset_dir=str(dataset_dir), 
        metadata_dir=str(metadata_dir), 
        output_dir=str(out_dir)
        )