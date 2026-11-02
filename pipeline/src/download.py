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

from dnastack import use
from dnastack import DataConnectClient

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def format_pipeline_step00_summary(
    *,
    output_dir: str
) -> str:
    summary = (
        "Pipeline Step00 summary:\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step00_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step00 Data Download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step00(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    factory = use('cloud.parkinsonsroadmap.org')
    data_connect_client: DataConnectClient = factory.get('data-connect-prod-team-edwards-pmdbs-spatial-geomx-th')

    # 1. Scarica le 4 tabelle
    edwards_data = pd.DataFrame(list(data_connect_client.query("SELECT * FROM \"collections\".\"prod_team_edwards_pmdbs_spatial_geomx_th\".\"edwards_data\"")))
    edwards_sample = pd.DataFrame(list(data_connect_client.query("SELECT * FROM \"collections\".\"prod_team_edwards_pmdbs_spatial_geomx_th\".\"edwards_sample\"")))
    edwards_subject = pd.DataFrame(list(data_connect_client.query("SELECT * FROM \"collections\".\"prod_team_edwards_pmdbs_spatial_geomx_th\".\"edwards_subject\"")))
    edwards_clinpath = pd.DataFrame(list(data_connect_client.query("SELECT * FROM \"collections\".\"prod_team_edwards_pmdbs_spatial_geomx_th\".\"edwards_clinpath\"")))

    # 1. Uniamo Data e Sample
    # Aggiungiamo 'replicate' e 'sample_id' (e le altre colonne in comune) per un match esatto!
    colonne_in_comune_ds = ['asap_dataset_id', 'asap_team_id', 'asap_sample_id', 'replicate', 'batch', 'sample_id', 'pool_id']
    
    # Per sicurezza, filtriamo solo le colonne che effettivamente esistono in ENTRAMBE le tabelle
    join_keys_1 = [col for col in colonne_in_comune_ds if col in edwards_data.columns and col in edwards_sample.columns]

    df_merged1 = pd.merge(
        edwards_data, 
        edwards_sample, 
        on=join_keys_1, 
        how='inner'
    )

    # 2. Uniamo con Subject
    colonne_in_comune_subj = ['asap_dataset_id', 'asap_team_id', 'asap_subject_id', 'subject_id']
    join_keys_2 = [col for col in colonne_in_comune_subj if col in df_merged1.columns and col in edwards_subject.columns]

    df_merged2 = pd.merge(
        df_merged1, 
        edwards_subject, 
        on=join_keys_2, 
        how='inner'
    )

    # 3. Uniamo con Clinpath
    colonne_in_comune_clin = ['asap_dataset_id', 'asap_team_id', 'asap_subject_id', 'subject_id']
    join_keys_3 = [col for col in colonne_in_comune_clin if col in df_merged2.columns and col in edwards_clinpath.columns]

    df_merged_final = pd.merge(
        df_merged2, 
        edwards_clinpath, 
        on=join_keys_3, 
        how='inner'
    )

    # Salva il risultato
    df_merged_final.to_csv(out_dir / "merged_data.csv", index=False)
    LOGGER.info("Dati scaricati e uniti con successo.")

    LOGGER.info(
        "\n%s",
        format_pipeline_step00_summary(
            output_dir=str(out_dir)
        ),
    )