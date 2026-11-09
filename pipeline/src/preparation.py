from __future__ import annotations

import argparse
import gc
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import os

try:
    from anndata.io import read_elem, write_elem
except ImportError:
    from anndata.experimental import read_elem, write_elem

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import (
    _ensure_list,
    _first_non_null,
    _overwrite_obs_in_h5ad,
    _sanitize_obs_for_h5ad,
    configure_logging,
    load_yaml_file,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_STEP01_CONFIG: dict[str, Any] = {
    "dpi_figures": 300,
    "chunk_size": 50000,
    "mapmycells_chunk_size": 2500,
    "mapmycells_batch_size": 200000,
    "mapmycells_max_gb": 6.0,
    "overwrite_full_dataset": False,
    "use_all_regions": False,
    "target_regions_level_1": [
        "Midbrain mesencephalon (MB, UBERON:0001891)",
        "Striatum (STR, UBERON:0002435)",
    ],
    "target_regions_level_2": [
        "Substantia nigra (SN, UBERON:0002038)",
        "Putamen (PUT, UBERON:0001874)",
    ],
    "target_conditions": ["Control", "PD"],
    "hmba_dopaminergic_subclass": "M Dopa",
    "hmba_dopaminergic_groups": [
        "SN SOX6 Dopa",
        "SN-VTR CALB1 Dopa",
        "SN-VTR GAD2 Dopa",
    ],
    "restrict_dopaminergic_to_sn": False,
    "min_bootstrapping_prob": 0.0,
    "mapmycells_normalization": "raw",  # "raw" or "log2CPM"
    "mapmycells_bootstrap_iteration": 100,
    "mapmycells_bootstrap_factor": 0.5,
    "mapmycells_n_processors": 3,
    "compute_umap_full": False,
    "compute_umap_dopaminergic": True,
}


def _resolve_step01_config(config_path: Path) -> dict[str, Any]:
    config = DEFAULT_STEP01_CONFIG.copy()
    if config_path.exists():
        loaded = load_yaml_file(config_path) or {}
        step01_cfg = loaded.get("step01", {})
        if step01_cfg and not isinstance(step01_cfg, dict):
            raise ValueError(f"Expected mapping under 'step01' in config: {config_path}")
        config.update({k: v for k, v in step01_cfg.items() if v is not None})
    else:
        LOGGER.warning("Config file not found (%s), using default values.", config_path)

    for list_key in (
        "target_regions_level_1",
        "target_regions_level_2",
        "target_conditions",
        "hmba_dopaminergic_groups",
    ):
        config[list_key] = [str(x) for x in _ensure_list(config[list_key])]

    config["chunk_size"] = max(1000, int(config["chunk_size"]))
    config["mapmycells_chunk_size"] = max(500, int(config.get("mapmycells_chunk_size", 2500)))
    config["mapmycells_batch_size"] = max(10000, int(config.get("mapmycells_batch_size", 200000)))
    config["use_all_regions"] = bool(config["use_all_regions"])
    return config


def match_regions(series: pd.Series, targets: list[str]) -> pd.Series:
    """Matches region strings either by exact match or by prefix before ' ('."""
    mask = pd.Series(False, index=series.index)
    if not targets:
        return mask
    mask |= series.isin(targets)
    for token in targets:
        prefix = token.split(" (")[0].strip()
        if prefix:
            mask |= series.str.contains(prefix, case=False, na=False)
    return mask


def _normalize_sample_id_index(raw_index: pd.Index) -> pd.Index:
    """
    Normalizes raw H5AD sample identifiers to match SAMPLE.csv ASAP_sample_id / sample_id:
      - Fixes typo 'ASAP_PMBDS_' -> 'ASAP_PMDBS_' (case-insensitive)
      - Strips replicate suffixes after sample number (e.g. '_s001_Rep1', '_s003_rep1', '_s001_1')
      - Strips trailing '_Rep1' / '_rep1' on non-ASAP sample IDs
    """
    return (
        raw_index.astype(str)
        .str.strip()
        .str.replace(r"(?i)^ASAP_PMBDS_", "ASAP_PMDBS_", regex=True)
        .str.replace(r"(?i)(_s\d+)_(?:rep\d+|\d+)$", r"\1", regex=True)
        .str.replace(r"(?i)_(?:rep\d+)$", "", regex=True)
    )


def build_enriched_sample_metadata(
    metadata_dir: Path,
) -> pd.DataFrame:
    """
    Loads and harmonizes SAMPLE.csv, SUBJECT.csv, CLINPATH.csv, and ASSAY.csv from metadata_dir.
    Handles multi-team duplicate subject rows cleanly using composite keys.
    """
    sample_path = metadata_dir / "SAMPLE.csv"
    subject_path = metadata_dir / "SUBJECT.csv"
    clinpath_path = metadata_dir / "CLINPATH.csv"
    assay_path = metadata_dir / "ASSAY.csv"

    if not sample_path.exists():
        raise FileNotFoundError(f"Required SAMPLE.csv not found in: {metadata_dir}")
    if not subject_path.exists():
        raise FileNotFoundError(f"Required SUBJECT.csv not found in: {metadata_dir}")

    sample_df = pd.read_csv(sample_path)
    subject_df = pd.read_csv(subject_path)

    subj_cols = [
        c
        for c in [
            "ASAP_dataset_id",
            "ASAP_subject_id",
            "sex",
            "race",
            "ethnicity",
            "biobank_name",
            "gp2_phenotype",
        ]
        if c in subject_df.columns
    ]
    subj_by_ds = (
        subject_df[subj_cols]
        .groupby(["ASAP_dataset_id", "ASAP_subject_id"], as_index=False)
        .agg(_first_non_null)
    )
    subj_by_id = (
        subject_df[[c for c in subj_cols if c != "ASAP_dataset_id"]]
        .groupby("ASAP_subject_id", as_index=False)
        .agg(_first_non_null)
    )

    merged = sample_df.merge(
        subj_by_ds,
        on=["ASAP_dataset_id", "ASAP_subject_id"],
        how="left",
    )

    for col in [c for c in subj_by_id.columns if c != "ASAP_subject_id"]:
        if col in merged.columns:
            fallback_map = subj_by_id.set_index("ASAP_subject_id")[col]
            merged[col] = merged[col].fillna(merged["ASAP_subject_id"].map(fallback_map))

    if clinpath_path.exists():
        clin_df = pd.read_csv(clinpath_path)
        clin_cols = [
            c
            for c in [
                "ASAP_dataset_id",
                "ASAP_subject_id",
                "age_at_death",
                "duration_pmi",
                "path_braak_asyn",
                "path_braak_nft",
                "hoehn_yahr_stage",
                "disease_duration_years",
                "primary_diagnosis",
                "cognitive_status",
                "sn_neuronal_loss",
            ]
            if c in clin_df.columns
        ]
        clin_dedup = (
            clin_df[clin_cols]
            .groupby(["ASAP_dataset_id", "ASAP_subject_id"], as_index=False)
            .agg(_first_non_null)
        )
        merged = merged.merge(
            clin_dedup,
            on=["ASAP_dataset_id", "ASAP_subject_id"],
            how="left",
        )

    if assay_path.exists():
        assay_df = pd.read_csv(assay_path)
        assay_cols = [
            c
            for c in [
                "ASAP_dataset_id",
                "ASAP_sample_id",
                "assay",
                "instrument",
                "RIN",
                "input_cell_count",
                "technology",
            ]
            if c in assay_df.columns
        ]
        assay_dedup = (
            assay_df[assay_cols]
            .groupby(["ASAP_dataset_id", "ASAP_sample_id"], as_index=False)
            .agg(_first_non_null)
        )
        merged = merged.merge(
            assay_dedup,
            on=["ASAP_dataset_id", "ASAP_sample_id"],
            how="left",
        )

    age_col = pd.to_numeric(merged.get("age_at_collection"), errors="coerce")
    if "age_at_death" in merged.columns:
        age_death = pd.to_numeric(merged["age_at_death"], errors="coerce")
        age_col = age_col.fillna(age_death)
    merged["age"] = age_col

    merged = merged.groupby("ASAP_sample_id", as_index=False).agg(_first_non_null)
    return merged


def _stream_csr_subset_to_h5ad(
    f_in: h5py.File,
    out_h5ad_path: Path,
    mask: np.ndarray,
    obs_filtered: pd.DataFrame,
    var_df: pd.DataFrame,
    chunk_size: int,
) -> None:
    """
    Streams rows where `mask` is True from `f_in['X']` directly into `out_h5ad_path['X']`
    one chunk at a time, storing data as float32 to halve disk and RAM usage.
    """
    n_total = int(len(mask))
    n_keep = int(mask.sum())
    n_vars = int(len(var_df))
    total_chunks = (n_total + chunk_size - 1) // chunk_size

    out_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    obs_clean = _sanitize_obs_for_h5ad(obs_filtered)
    var_clean = _sanitize_obs_for_h5ad(var_df)

    x_node = f_in["X"]
    encoding_type = x_node.attrs.get("encoding-type", "")

    LOGGER.info(
        "Streaming %d / %d selected cells to %s in %d chunks (chunk_size=%d)...",
        n_keep,
        n_total,
        out_h5ad_path,
        total_chunks,
        chunk_size,
    )

    with h5py.File(out_h5ad_path, "w") as f_out:
        f_out.attrs["encoding-type"] = "anndata"
        f_out.attrs["encoding-version"] = "0.1.0"

        write_elem(f_out, "obs", obs_clean)
        write_elem(f_out, "var", var_clean)

        if isinstance(x_node, h5py.Group) and encoding_type == "csr_matrix":
            in_data = x_node["data"]
            in_indices = x_node["indices"]
            in_indptr = x_node["indptr"][:]

            g_x = f_out.create_group("X")
            g_x.attrs["encoding-type"] = "csr_matrix"
            g_x.attrs["encoding-version"] = "0.1.0"
            g_x.attrs["shape"] = np.array([n_keep, n_vars], dtype=np.int64)

            d_data = g_x.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
                chunks=(min(1000000, max(65536, chunk_size * 20)),),
            )
            d_indices = g_x.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                dtype=in_indices.dtype,
                chunks=(min(1000000, max(65536, chunk_size * 20)),),
            )

            out_indptr = np.zeros(n_keep + 1, dtype=np.int64)
            kept_so_far = 0
            nnz_so_far = 0

            for chunk_idx, start in enumerate(range(0, n_total, chunk_size), start=1):
                end = min(start + chunk_size, n_total)
                sub_mask = mask[start:end]

                if not np.any(sub_mask):
                    continue

                val_start = int(in_indptr[start])
                val_end = int(in_indptr[end])

                data_block = in_data[val_start:val_end]
                indices_block = in_indices[val_start:val_end]
                indptr_block = in_indptr[start : end + 1] - val_start

                block_csr = sp.csr_matrix(
                    (data_block, indices_block, indptr_block),
                    shape=(end - start, n_vars),
                )
                sub_csr = block_csr[sub_mask]
                sub_csr.sort_indices()

                n_sub_rows = sub_csr.shape[0]
                n_sub_nnz = int(sub_csr.nnz)

                if n_sub_nnz > 0:
                    d_data.resize((nnz_so_far + n_sub_nnz,))
                    d_data[nnz_so_far : nnz_so_far + n_sub_nnz] = sub_csr.data.astype(
                        np.float32, copy=False
                    )

                    d_indices.resize((nnz_so_far + n_sub_nnz,))
                    d_indices[nnz_so_far : nnz_so_far + n_sub_nnz] = sub_csr.indices

                out_indptr[kept_so_far + 1 : kept_so_far + 1 + n_sub_rows] = (
                    sub_csr.indptr[1:] + nnz_so_far
                )

                kept_so_far += n_sub_rows
                nnz_so_far += n_sub_nnz

                del data_block, indices_block, indptr_block, block_csr, sub_csr

                if chunk_idx % 10 == 0 or end == n_total:
                    LOGGER.info(
                        "  Streamed chunk %d / %d (kept %d / %d cells, nnz=%d)",
                        chunk_idx,
                        total_chunks,
                        kept_so_far,
                        n_keep,
                        nnz_so_far,
                    )

            g_x.create_dataset("indptr", data=out_indptr, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported X encoding-type: {encoding_type}")

    gc.collect()


def _load_subset_adata_from_h5ad_in_chunks(
    h5ad_path: Path,
    mask: np.ndarray,
    obs_filtered: pd.DataFrame,
    chunk_size: int,
) -> ad.AnnData:
    """
    Loads a small subset of cells (e.g., dopaminergic neurons) into an in-memory AnnData
    by streaming contiguous chunks from `h5ad_path` via h5py.
    """
    n_total = int(len(mask))
    n_keep = int(mask.sum())

    with h5py.File(h5ad_path, "r") as f_in:
        var_df = read_elem(f_in["var"])
        n_vars = int(len(var_df))

        if n_keep == 0:
            empty_x = sp.csr_matrix((0, n_vars), dtype=np.float32)
            return ad.AnnData(X=empty_x, obs=obs_filtered.copy(), var=var_df)

        x_node = f_in["X"]
        in_data = x_node["data"]
        in_indices = x_node["indices"]
        in_indptr = x_node["indptr"][:]

        chunks: list[sp.csr_matrix] = []
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            sub_mask = mask[start:end]
            if not np.any(sub_mask):
                continue

            val_start = int(in_indptr[start])
            val_end = int(in_indptr[end])
            block_csr = sp.csr_matrix(
                (
                    in_data[val_start:val_end],
                    in_indices[val_start:val_end],
                    in_indptr[start : end + 1] - val_start,
                ),
                shape=(end - start, n_vars),
            )
            chunks.append(block_csr[sub_mask])

    x_combined = (
        sp.vstack(chunks, format="csr")
        if chunks
        else sp.csr_matrix((0, n_vars), dtype=np.float32)
    )
    del chunks
    gc.collect()

    return ad.AnnData(X=x_combined, obs=obs_filtered.copy(), var=var_df)


def stream_enrich_and_filter_raw_h5ad(
    raw_h5ad_path: Path,
    full_out_path: Path,
    sample_meta: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Opens `raw_h5ad_path` via h5py (without loading layers/obsp/obsm/X into RAM),
    normalizes sample IDs, filters cells by region and condition, maps metadata onto
    the retained cells, and streams the filtered expression matrix directly to `full_out_path`.
    """
    chunk_size = int(config["chunk_size"])
    use_all_regions = bool(config.get("use_all_regions", False))
    target_r1: list[str] = config["target_regions_level_1"]
    target_r2: list[str] = config["target_regions_level_2"]
    target_conds: list[str] = config["target_conditions"]

    r1_meta = sample_meta.get("region_level_1", pd.Series("", index=sample_meta.index)).astype(str)
    r2_meta = sample_meta.get("region_level_2", pd.Series("", index=sample_meta.index)).astype(str)
    cond_meta = sample_meta.get("condition_id", pd.Series("", index=sample_meta.index)).astype(str)

    if use_all_regions:
        region_meta_mask = pd.Series(True, index=sample_meta.index)
    else:
        region_meta_mask = (
            match_regions(r1_meta, target_r1)
            | match_regions(r2_meta, target_r2)
        )

    cond_meta_mask = (
        cond_meta.isin(target_conds)
        if target_conds
        else pd.Series(True, index=sample_meta.index)
    )
    valid_meta = sample_meta.loc[region_meta_mask & cond_meta_mask].copy()

    with h5py.File(raw_h5ad_path, "r") as f_in:
        LOGGER.info("Reading /obs and /var metadata from %s via h5py...", raw_h5ad_path)
        obs = read_elem(f_in["obs"])
        var_df = read_elem(f_in["var"])
        initial_cells = int(len(obs))

        obs_join_col = None
        meta_join_col = "ASAP_sample_id"
        for candidate in ["ASAP_sample_id", "sample", "sample_id", "ASAP_subject_id", "subject_id"]:
            if candidate in obs.columns:
                obs_join_col = candidate
                break

        if not obs_join_col:
            raise KeyError(
                "Unable to locate a sample/subject identifier column in raw h5ad /obs "
                f"(available columns: {list(obs.columns)})"
            )

        valid_asap_ids = set(
            _normalize_sample_id_index(pd.Index(valid_meta["ASAP_sample_id"].dropna()))
        )
        valid_ids = set(valid_asap_ids)

        if "sample_id" in valid_meta.columns:
            valid_ids |= set(
                _normalize_sample_id_index(pd.Index(valid_meta["sample_id"].dropna()))
            )

        if obs_join_col in {"ASAP_subject_id", "subject_id"} and "ASAP_subject_id" in valid_meta.columns:
            valid_ids = set(
                valid_meta["ASAP_subject_id"].dropna().astype(str).str.strip()
            )
            meta_join_col = "ASAP_subject_id"

        unique_raw_vals = pd.Index(obs[obs_join_col].dropna().unique())
        unique_norm_vals = _normalize_sample_id_index(unique_raw_vals)
        raw_to_norm_map = dict(zip(unique_raw_vals, unique_norm_vals))

        matching_raw_set = set(unique_raw_vals[unique_norm_vals.isin(valid_ids)])

        LOGGER.info("========== SAMPLE ID NORMALIZATION & MATCHING ==========")
        LOGGER.info("Selected OBS join column: %s -> %s", obs_join_col, meta_join_col)
        LOGGER.info("Valid metadata samples (after region/condition filter): %d", len(valid_meta))
        LOGGER.info("Unique raw sample IDs in H5AD: %d", len(unique_raw_vals))
        LOGGER.info(
            "Matched unique sample IDs (after normalization): %d / %d",
            len(matching_raw_set),
            len(valid_meta),
        )

        final_mask_np = obs[obs_join_col].isin(matching_raw_set).to_numpy(dtype=bool)
        n_matches = int(final_mask_np.sum())

        LOGGER.info(
            "Matching cells: %d / %d (use_all_regions=%s)",
            n_matches,
            initial_cells,
            use_all_regions,
        )

        if n_matches == 0:
            raise ValueError(
                f"Zero matching cells after ID normalization. Join column: {obs_join_col}; "
                f"Valid IDs: {len(valid_ids)}."
            )

        obs_filtered = obs.iloc[final_mask_np].copy()
        del obs
        gc.collect()

        norm_key_series = obs_filtered[obs_join_col].map(raw_to_norm_map).astype(str)
        obs_filtered["ASAP_sample_id"] = norm_key_series

        lookup_key_col = "ASAP_sample_id"
        if (
            "sample_id" in sample_meta.columns
            and not norm_key_series.isin(sample_meta["ASAP_sample_id"].astype(str)).any()
        ):
            lookup_key_col = "sample_id"
        elif meta_join_col == "ASAP_subject_id":
            lookup_key_col = "ASAP_subject_id"

        meta_dedup = sample_meta.drop_duplicates(subset=[lookup_key_col]).copy()
        meta_dedup["_norm_key"] = _normalize_sample_id_index(pd.Index(meta_dedup[lookup_key_col]))
        meta_indexed = meta_dedup.drop_duplicates(subset=["_norm_key"]).set_index("_norm_key")

        cols_to_map = [
            "ASAP_sample_id",
            "ASAP_subject_id",
            "ASAP_dataset_id",
            "ASAP_team_id",
            "sex",
            "age",
            "condition_id",
            "region_level_1",
            "region_level_2",
            "region_level_3",
            "duration_pmi",
            "path_braak_asyn",
            "path_braak_nft",
            "hoehn_yahr_stage",
            "disease_duration_years",
            "race",
            "biobank_name",
            "assay",
            "instrument",
            "RIN",
            "input_cell_count",
            "technology",
        ]
        for col in cols_to_map:
            if col in meta_indexed.columns:
                obs_filtered[col] = norm_key_series.map(meta_indexed[col]).to_numpy()

        _stream_csr_subset_to_h5ad(
            f_in=f_in,
            out_h5ad_path=full_out_path,
            mask=final_mask_np,
            obs_filtered=obs_filtered,
            var_df=var_df,
            chunk_size=chunk_size,
        )

    filter_stats = {
        "initial_cells_raw": initial_cells,
        "use_all_regions": use_all_regions,
        "chunk_size": chunk_size,
        "retained_cells_full_dataset": int(len(obs_filtered)),
        "retained_samples_full_dataset": (
            int(obs_filtered["ASAP_sample_id"].nunique())
            if "ASAP_sample_id" in obs_filtered.columns
            else None
        ),
        "retained_subjects_full_dataset": (
            int(obs_filtered["ASAP_subject_id"].nunique())
            if "ASAP_subject_id" in obs_filtered.columns
            else None
        ),
        "cells_by_region_level_2": (
            {
                str(k): int(v)
                for k, v in obs_filtered["region_level_2"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            }
            if "region_level_2" in obs_filtered.columns
            else {}
        ),
    }
    return obs_filtered, filter_stats


def _compute_umap_if_needed(adata: ad.AnnData, force: bool = False) -> None:
    """Computes PCA, neighborhood graph, and UMAP coordinates on a temporary normalized copy."""
    if not force and "X_umap" in adata.obsm:
        return
    if adata.n_obs < 5:
        return
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    n_comps = max(2, min(30, adata_tmp.n_obs - 1, adata_tmp.n_vars - 1))
    sc.pp.pca(adata_tmp, n_comps=n_comps)
    sc.pp.neighbors(adata_tmp)
    sc.tl.umap(adata_tmp)
    adata.obsm["X_umap"] = adata_tmp.obsm["X_umap"]


def _prepare_var_for_mapmycells(full_out_path: Path) -> tuple[pd.DataFrame, str | None, bool]:
    """
    Checks `/var` in `full_out_path` for Ensembl IDs (`ENSG...`) and returns
    `(var_df_clean, query_gene_id_col, map_to_ensembl)`.
    """
    with h5py.File(full_out_path, "r") as f:
        var_df = read_elem(f["var"])

    idx_str = pd.Index(var_df.index.astype(str))
    if idx_str.str.startswith("ENSG").any():
        if idx_str.str.contains(r"\.\d+$", regex=True).any():
            var_df["_ensembl_id_clean"] = idx_str.str.replace(r"\.\d+$", "", regex=True).to_numpy()
            var_clean = _sanitize_obs_for_h5ad(var_df)
            with h5py.File(full_out_path, "a") as f:
                write_elem(f, "var", var_clean)
            return var_clean, "_ensembl_id_clean", False
        return _sanitize_obs_for_h5ad(var_df), None, False

    for col in var_df.columns:
        s_col = var_df[col].astype(str)
        if s_col.str.startswith("ENSG").any():
            LOGGER.info(
                "Found Ensembl IDs in var['%s']; preparing clean column '_ensembl_id_clean' for MapMyCells.",
                col,
            )
            var_df["_ensembl_id_clean"] = (
                s_col.str.strip().str.replace(r"\.\d+$", "", regex=True).to_numpy()
            )
            var_clean = _sanitize_obs_for_h5ad(var_df)
            with h5py.File(full_out_path, "a") as f:
                write_elem(f, "var", var_clean)
            return var_clean, "_ensembl_id_clean", False

    LOGGER.info(
        "No Ensembl ID column found in /var (columns=%s); enabling map_to_ensembl=True in MapMyCells.",
        list(var_df.columns),
    )
    return _sanitize_obs_for_h5ad(var_df), None, True


def _write_minimal_shard_h5ad(
    full_out_path: Path,
    shard_h5ad_path: Path,
    start_row: int,
    end_row: int,
    var_clean: pd.DataFrame,
    obs_names_slice: pd.Index,
) -> None:
    """
    Extracts a contiguous slice of rows [start_row:end_row] from `full_out_path['X']`
    into a lightweight `shard_h5ad_path` (with float32 data and minimal obs) for MapMyCells.
    """
    n_rows = end_row - start_row
    n_vars = int(len(var_clean))
    minimal_obs = pd.DataFrame(index=obs_names_slice.astype(str))
    minimal_obs.index.name = "cell_id"

    with h5py.File(full_out_path, "r") as f_in, h5py.File(shard_h5ad_path, "w") as f_out:
        f_out.attrs["encoding-type"] = "anndata"
        f_out.attrs["encoding-version"] = "0.1.0"
        write_elem(f_out, "obs", minimal_obs)
        write_elem(f_out, "var", var_clean)

        x_in = f_in["X"]
        in_indptr = x_in["indptr"][start_row : end_row + 1]
        val_start = int(in_indptr[0])
        val_end = int(in_indptr[-1])

        data_slice = x_in["data"][val_start:val_end].astype(np.float32, copy=False)
        indices_slice = x_in["indices"][val_start:val_end]
        indptr_slice = (in_indptr - val_start).astype(np.int64)

        g_x = f_out.create_group("X")
        g_x.attrs["encoding-type"] = "csr_matrix"
        g_x.attrs["encoding-version"] = "0.1.0"
        g_x.attrs["shape"] = np.array([n_rows, n_vars], dtype=np.int64)
        g_x.create_dataset("data", data=data_slice, dtype=np.float32)
        g_x.create_dataset("indices", data=indices_slice, dtype=indices_slice.dtype)
        g_x.create_dataset("indptr", data=indptr_slice, dtype=np.int64)


def _run_mapmycells_in_checkpointed_batches(
    full_out_path: Path,
    obs_index: pd.Index,
    stats_path: Path,
    markers_path: Path,
    resolved_csv_path: Path,
    config: dict[str, Any],
    tmp_dir_raw: Path | str | None = None,
) -> None:
    """
    Executes Allen Institute cell_type_mapper in checkpointed batches of `mapmycells_batch_size`
    cells. Each batch writes its own CSV on disk (`hmba_shards/hmba_batch_XXX.csv`) and frees RAM
    and `/tmp` immediately, preventing OOM crashes and allowing instant resume if interrupted.
    """
    try:
        from cell_type_mapper.cli.from_specified_markers import FromSpecifiedMarkersRunner
    except ImportError as exc:
        raise ImportError(
            "Package 'cell_type_mapper' is required for --run-mapmycells. "
            "Install it via 'poetry add git+https://github.com/AllenInstitute/cell_type_mapper.git' "
            "or provide --hmba-mapping-csv."
        ) from exc

    n_total = int(len(obs_index))
    batch_size = int(config.get("mapmycells_batch_size", 200000))
    mapmycells_chunk_size = int(config.get("mapmycells_chunk_size", 2500))
    max_gb = float(config.get("mapmycells_max_gb", 6.0))
    n_batches = (n_total + batch_size - 1) // batch_size

    var_clean, query_gene_id_col, map_to_ensembl = _prepare_var_for_mapmycells(full_out_path)

    shards_dir = resolved_csv_path.parent / "hmba_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)

    shard_csv_paths: list[Path] = []

    for batch_idx, start_row in enumerate(range(0, n_total, batch_size), start=1):
        end_row = min(start_row + batch_size, n_total)
        shard_csv = shards_dir / f"hmba_batch_{batch_idx:03d}_{start_row}_{end_row}.csv"
        shard_csv_paths.append(shard_csv)

        # Resume check: if this batch CSV already exists and has all expected rows, skip it!
        if shard_csv.exists() and shard_csv.stat().st_size > 100:
            try:
                existing_df = pd.read_csv(shard_csv, comment="#", usecols=[0])
                if len(existing_df) == (end_row - start_row):
                    LOGGER.info(
                        "[MapMyCells Batch %d/%d] Already completed on disk (%d cells): %s -> Skipping!",
                        batch_idx,
                        n_batches,
                        len(existing_df),
                        shard_csv.name,
                    )
                    del existing_df
                    continue
            except Exception:
                pass

        LOGGER.info(
            "[MapMyCells Batch %d/%d] Preparing shard for cells %d..%d (%d cells)...",
            batch_idx,
            n_batches,
            start_row,
            end_row,
            end_row - start_row,
        )

        # Use a fresh temporary directory per batch so /tmp is completely wiped after every batch
        if tmp_dir_raw:
            base_tmp = Path(tmp_dir_raw)
            base_tmp.mkdir(parents=True, exist_ok=True)
            tmp_ctx = tempfile.TemporaryDirectory(dir=str(base_tmp))
        else:
            tmp_ctx = tempfile.TemporaryDirectory()

        with tmp_ctx as batch_tmp_dir:
            batch_tmp_path = Path(batch_tmp_dir)
            shard_h5ad = batch_tmp_path / f"query_shard_{batch_idx:03d}.h5ad"
            shard_json = batch_tmp_path / f"extended_shard_{batch_idx:03d}.json"

            _write_minimal_shard_h5ad(
                full_out_path=full_out_path,
                shard_h5ad_path=shard_h5ad,
                start_row=start_row,
                end_row=end_row,
                var_clean=var_clean,
                obs_names_slice=obs_index[start_row:end_row],
            )

            runner_config: dict[str, Any] = {
                "query_path": str(shard_h5ad),
                "extended_result_path": str(shard_json),
                "csv_result_path": str(shard_csv),
                "precomputed_stats": {"path": str(stats_path)},
                "query_markers": {"serialized_lookup": str(markers_path)},
                "query_gene_id_col": query_gene_id_col,
                "map_to_ensembl": map_to_ensembl,
                "max_gb": max_gb,
                "type_assignment": {
                    "normalization": str(config["mapmycells_normalization"]),
                    "bootstrap_iteration": int(config["mapmycells_bootstrap_iteration"]),
                    "bootstrap_factor": float(config["mapmycells_bootstrap_factor"]),
                    "n_processors": int(config["mapmycells_n_processors"]),
                    "chunk_size": mapmycells_chunk_size,
                    "n_runners_up": 0,
                },
                "tmp_dir": None,
            }

            LOGGER.info(
                "[MapMyCells Batch %d/%d] Running cell_type_mapper (chunk_size=%d, n_processors=%d)...",
                batch_idx,
                n_batches,
                mapmycells_chunk_size,
                int(config["mapmycells_n_processors"]),
            )
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["AIBS_BKP_USE_TORCH"] = "true"
            runner = FromSpecifiedMarkersRunner(args=[], input_data=runner_config)
            runner.run()
            del runner

        gc.collect()
        LOGGER.info("[MapMyCells Batch %d/%d] Saved checkpoint: %s", batch_idx, n_batches, shard_csv)

    # Concatenate all batch CSVs into the final resolved_csv_path
    LOGGER.info("Combining %d MapMyCells batch CSVs into %s...", len(shard_csv_paths), resolved_csv_path)
    dfs = [pd.read_csv(p, comment="#") for p in shard_csv_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(resolved_csv_path, index=False)
    del dfs, combined_df
    gc.collect()
    LOGGER.info("Unified MapMyCells CSV saved: %s", resolved_csv_path)


def run_or_apply_hmba_basal_ganglia_mapping(
    obs_full: pd.DataFrame,
    full_out_path: Path | str,
    config: dict[str, Any],
    hmba_mapping_csv: Path | str | None = None,
    hmba_extended_json: Path | str | None = None,
    run_mapmycells: bool = False,
    precomputed_stats_path: Path | str | None = None,
    query_markers_path: Path | str | None = None,
    mapmycells_tmp_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Runs or applies the Allen Institute HMBA Basal Ganglia Consensus Taxonomy (CCN20250428)
    using checkpointed batches and updates `/obs` in `full_out_path`.
    """
    full_out_path = Path(full_out_path)
    chunk_size = int(config["chunk_size"])

    csv_path_raw = hmba_mapping_csv or config.get("hmba_mapping_csv")
    stats_path_raw = precomputed_stats_path or config.get("precomputed_stats_path")
    markers_path_raw = query_markers_path or config.get("query_markers_path")
    tmp_dir_raw = mapmycells_tmp_dir or config.get("mapmycells_tmp_dir")

    resolved_csv_path = (
        Path(csv_path_raw)
        if csv_path_raw
        else full_out_path.parent / "hmba_mapmycells_results.csv"
    )

    if run_mapmycells:
        if not stats_path_raw or not markers_path_raw:
            raise ValueError(
                "To run MapMyCells locally (--run-mapmycells), both --precomputed-stats "
                "and --query-markers paths must be provided via CLI or config."
            )

        stats_path = Path(stats_path_raw)
        markers_path = Path(markers_path_raw)
        if not stats_path.exists():
            raise FileNotFoundError(f"HMBA precomputed_stats file not found: {stats_path}")
        if not markers_path.exists():
            raise FileNotFoundError(f"HMBA query_markers file not found: {markers_path}")

        # Keep only the cell_id index in RAM while running MapMyCells workers,
        # freeing the ~3 GB obs_full DataFrame so forked processes stay lightweight!
        obs_index_copy = pd.Index(obs_full.index.astype(str))
        del obs_full
        gc.collect()

        _run_mapmycells_in_checkpointed_batches(
            full_out_path=full_out_path,
            obs_index=obs_index_copy,
            stats_path=stats_path,
            markers_path=markers_path,
            resolved_csv_path=resolved_csv_path,
            config=config,
            tmp_dir_raw=tmp_dir_raw,
        )

        # Reload obs_full from full_out_path after MapMyCells completes
        with h5py.File(full_out_path, "r") as f_reload:
            obs_full = read_elem(f_reload["obs"])

    if (run_mapmycells or csv_path_raw is not None) and resolved_csv_path.exists():
        LOGGER.info("Applying HMBA MapMyCells annotations from %s...", resolved_csv_path)
        map_df = pd.read_csv(resolved_csv_path, comment="#")
        cell_id_col = "cell_id" if "cell_id" in map_df.columns else map_df.columns[0]
        map_df[cell_id_col] = map_df[cell_id_col].astype(str)
        map_df = map_df.drop_duplicates(subset=[cell_id_col]).set_index(cell_id_col)

        rename_dict: dict[str, str] = {}
        for col in map_df.columns:
            col_lower = col.lower()
            for level in ("Neighborhood", "Class", "Subclass", "Group", "Cluster"):
                lvl_lower = level.lower()
                if col_lower == f"{lvl_lower}_name" or col_lower.endswith(f"_{lvl_lower}_name"):
                    rename_dict[col] = f"HMBA_{level}_name"
                elif col_lower == f"{lvl_lower}_label" or col_lower.endswith(f"_{lvl_lower}_label"):
                    rename_dict[col] = f"HMBA_{level}_label"
                elif col_lower == f"{lvl_lower}_bootstrapping_probability" or col_lower.endswith(
                    f"_{lvl_lower}_bootstrapping_probability"
                ):
                    rename_dict[col] = f"HMBA_{level}_bootstrapping_probability"
        map_df = map_df.rename(columns=rename_dict)

        obs_index_str = obs_full.index.astype(str)
        for col in map_df.columns:
            obs_full[col] = map_df[col].reindex(obs_index_str).to_numpy()
        del map_df
        gc.collect()

    for level in ("Neighborhood", "Class", "Subclass", "Group", "Cluster"):
        raw_col = f"{level}_name"
        hmba_col = f"HMBA_{level}_name"
        if hmba_col not in obs_full.columns and raw_col in obs_full.columns:
            obs_full[hmba_col] = obs_full[raw_col]

    if "HMBA_Group_name" not in obs_full.columns:
        raise KeyError(
            "Column 'HMBA_Group_name' not found in obs after HMBA mapping step. "
            "Provide --hmba-mapping-csv or run with --run-mapmycells."
        )

    min_prob = float(config.get("min_bootstrapping_prob", 0.0))
    prob_col = "HMBA_Group_bootstrapping_probability"
    if min_prob > 0.0 and prob_col in obs_full.columns:
        keep_prob = (
            pd.to_numeric(obs_full[prob_col], errors="coerce").fillna(0.0) >= min_prob
        ).to_numpy(dtype=bool)
        LOGGER.info(
            "Filtering full_out_path cells with %s >= %.2f: %d / %d kept.",
            prob_col,
            min_prob,
            int(keep_prob.sum()),
            len(obs_full),
        )
        obs_full = obs_full.iloc[keep_prob].copy()
        tmp_filtered = full_out_path.with_suffix(".prob_tmp.h5ad")
        with h5py.File(full_out_path, "r") as f_in:
            var_df = read_elem(f_in["var"])
            _stream_csr_subset_to_h5ad(
                f_in=f_in,
                out_h5ad_path=tmp_filtered,
                mask=keep_prob,
                obs_filtered=obs_full,
                var_df=var_df,
                chunk_size=chunk_size,
            )
        tmp_filtered.replace(full_out_path)
    else:
        LOGGER.info("Updating /obs with HMBA annotations in-place on %s...", full_out_path)
        _overwrite_obs_in_h5ad(full_out_path, obs_full)

    group_counts = obs_full["HMBA_Group_name"].value_counts(dropna=False).to_dict()
    subclass_counts = (
        obs_full["HMBA_Subclass_name"].value_counts(dropna=False).to_dict()
        if "HMBA_Subclass_name" in obs_full.columns
        else {}
    )
    hmba_stats = {
        "mapping_csv_used": (
            str(resolved_csv_path)
            if (run_mapmycells or csv_path_raw is not None) and resolved_csv_path.exists()
            else "pre_annotated_in_h5ad"
        ),
        "total_hmba_groups_detected": int(obs_full["HMBA_Group_name"].nunique(dropna=True)),
        "hmba_subclass_counts_full_dataset": {str(k): int(v) for k, v in subclass_counts.items()},
        "hmba_group_counts_full_dataset": {str(k): int(v) for k, v in group_counts.items()},
    }
    return obs_full, hmba_stats


def isolate_dopaminergic_compartment_from_disk(
    full_out_path: Path,
    obs_full: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Streams only the midbrain dopaminergic populations (SN SOX6 Dopa, SN-VTR CALB1 Dopa,
    SN-VTR GAD2 Dopa) from `full_out_path` on disk into a compact AnnData object.
    """
    dopa_groups: list[str] = config["hmba_dopaminergic_groups"]
    dopa_subclass = str(config["hmba_dopaminergic_subclass"])
    restrict_to_sn = bool(config["restrict_dopaminergic_to_sn"])
    chunk_size = int(config["chunk_size"])

    group_s = obs_full["HMBA_Group_name"].astype(str)
    mask = group_s.isin(dopa_groups)

    if "HMBA_Subclass_name" in obs_full.columns and dopa_subclass:
        subclass_s = obs_full["HMBA_Subclass_name"].astype(str)
        mask = mask & (subclass_s == dopa_subclass)

    if restrict_to_sn and "region_level_2" in obs_full.columns:
        r2_s = obs_full["region_level_2"].astype(str)
        mask = mask & r2_s.str.contains("Substantia nigra", case=False, na=False)

    mask_np = mask.to_numpy(dtype=bool)
    obs_dopa = _sanitize_obs_for_h5ad(obs_full.iloc[mask_np].copy())

    LOGGER.info(
        "Streaming %d dopaminergic cells from %s...",
        int(mask_np.sum()),
        full_out_path,
    )
    adata_dopa = _load_subset_adata_from_h5ad_in_chunks(
        h5ad_path=full_out_path,
        mask=mask_np,
        obs_filtered=obs_dopa,
        chunk_size=chunk_size,
    )

    if bool(config.get("compute_umap_dopaminergic", True)) and adata_dopa.n_obs >= 5:
        LOGGER.info("Computing UMAP embedding for the isolated dopaminergic dataset...")
        _compute_umap_if_needed(adata_dopa)

    dopa_stats = {
        "dopaminergic_cells": int(adata_dopa.n_obs),
        "dopaminergic_subjects": (
            int(adata_dopa.obs["ASAP_subject_id"].nunique())
            if "ASAP_subject_id" in adata_dopa.obs.columns
            else None
        ),
        "restrict_dopaminergic_to_sn": restrict_to_sn,
        "counts_by_hmba_group": {
            str(k): int(v)
            for k, v in adata_dopa.obs["HMBA_Group_name"].value_counts().to_dict().items()
        },
        "counts_by_region_level_2": (
            {
                str(k): int(v)
                for k, v in adata_dopa.obs["region_level_2"].value_counts().to_dict().items()
            }
            if "region_level_2" in adata_dopa.obs.columns
            else {}
        ),
    }
    return adata_dopa, dopa_stats


def save_step01_qc_plots(
    obs_full: pd.DataFrame,
    adata_dopa: ad.AnnData,
    plots_dir: Path,
    config: dict[str, Any],
) -> dict[str, str]:
    """Saves regional cohort overview, full HMBA groups distribution, and dopaminergic subclusters."""
    dpi = int(config["dpi_figures"])
    saved: dict[str, str] = {}

    if "region_level_2" in obs_full.columns and "condition_id" in obs_full.columns:
        df_subj = (
            obs_full[["ASAP_sample_id", "region_level_2", "condition_id"]]
            .drop_duplicates()
            .copy()
        )
        ct = pd.crosstab(df_subj["region_level_2"], df_subj["condition_id"])
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
            ct.plot(kind="bar", ax=ax, rot=25, edgecolor="black", width=0.65)
            ax.set_title("Step 01 Retained Samples by Brain Region and Condition")
            ax.set_xlabel("Brain Region (region_level_2)")
            ax.set_ylabel("Number of Samples")
            ax.legend(title="Condition", frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for container in ax.containers:
                ax.bar_label(container, padding=3)
            region_plot = plots_dir / "step01_regional_samples_summary.png"
            fig.savefig(region_plot, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["regional_samples_plot"] = str(region_plot)

    if "HMBA_Group_name" in obs_full.columns and len(obs_full) > 0:
        top_groups = obs_full["HMBA_Group_name"].value_counts().head(25).iloc[::-1]
        if not top_groups.empty:
            fig, ax = plt.subplots(figsize=(9, max(5, len(top_groups) * 0.32)), facecolor="white")
            top_groups.plot(kind="barh", ax=ax, color="#4c72b0", edgecolor="black")
            ax.set_title("Step 01 Full Dataset: Top HMBA Basal Ganglia Groups")
            ax.set_xlabel("Number of Nuclei")
            ax.set_ylabel("HMBA_Group_name")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            full_groups_plot = plots_dir / "step01_full_dataset_hmba_groups.png"
            fig.savefig(full_groups_plot, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved["full_dataset_hmba_groups_plot"] = str(full_groups_plot)

    if adata_dopa.n_obs > 0:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        grp_counts = adata_dopa.obs["HMBA_Group_name"].value_counts()
        grp_counts.plot(
            kind="bar",
            ax=ax,
            color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            edgecolor="black",
            rot=15,
        )
        ax.set_title("Step 01 Isolated HMBA Dopaminergic Subpopulations (Cells)")
        ax.set_xlabel("HMBA_Group_name")
        ax.set_ylabel("Number of Nuclei")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for container in ax.containers:
            ax.bar_label(container, padding=3)
        dopa_plot = plots_dir / "step01_dopaminergic_groups_counts.png"
        fig.savefig(dopa_plot, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved["dopaminergic_groups_plot"] = str(dopa_plot)

    return saved


def format_pipeline_step01_summary(
    *,
    raw_h5ad_path: str,
    metadata_dir: str,
    use_all_regions: bool,
    chunk_size: int,
    full_dataset_path: str,
    dopaminergic_dataset_path: str,
    report_path: str,
) -> None:
    LOGGER.info(
        "Pipeline Step01 summary:\n"
        f"- Raw Input Dataset (.h5ad): {raw_h5ad_path}\n"
        f"- Metadata CSV Directory: {metadata_dir}\n"
        f"- Use All Brain Regions: {use_all_regions} (chunk_size={chunk_size})\n"
        f"- Full Annotated Dataset (all HMBA groups): {full_dataset_path}\n"
        f"- Dopaminergic Dataset (only Dopa neurons): {dopaminergic_dataset_path}\n"
        f"- Report Path: {report_path}"
    )


def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step01 Out-of-Core Metadata Enrichment, Chunked Regional Filtering, "
            "Batched & Checkpointed HMBA Basal Ganglia Mapping, and Export of Full & Dopaminergic Datasets"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    raw_dir = default_repo_root / "pipeline" / "data" / "raw"
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    plots_dir = default_repo_root / "pipeline" / "plots"
    reports_dir = default_repo_root / "pipeline" / "reports"

    parser.add_argument(
        "--config",
        default=str(default_repo_root / "pipeline" / "config" / "step01.yml"),
        help="YAML config file for step01 tunables.",
    )
    parser.add_argument(
        "--raw-h5ad",
        default=str(raw_dir / "asap-cohort.merged_cleaned_unfiltered.h5ad"),
        help="Path to the raw merged unfiltered .h5ad file.",
    )
    parser.add_argument(
        "--metadata-dir",
        default=str(raw_dir / "metadata"),
        help="Directory containing SAMPLE.csv, SUBJECT.csv, CLINPATH.csv, and ASSAY.csv.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of cells per chunk when streaming the .h5ad file (overrides config).",
    )
    parser.add_argument(
        "--mapmycells-chunk-size",
        type=int,
        default=None,
        help="Number of cells per dense worker chunk in MapMyCells (default: 2500).",
    )
    parser.add_argument(
        "--mapmycells-batch-size",
        type=int,
        default=None,
        help="Number of cells per checkpointed batch in MapMyCells (default: 200000).",
    )
    parser.add_argument(
        "--overwrite-full-dataset",
        action="store_true",
        help="Force re-streaming full_dataset.h5ad from raw H5AD even if it already exists on disk.",
    )
    parser.add_argument(
        "--use-all-regions",
        action="store_true",
        help="If set, retains all brain regions instead of filtering by target_regions_level_1/2.",
    )
    parser.add_argument(
        "--hmba-mapping-csv",
        default=None,
        help="Path to the CSV output generated by Allen Institute MapMyCells (HMBA Basal Ganglia).",
    )
    parser.add_argument(
        "--run-mapmycells",
        action="store_true",
        help="Run Allen Institute cell_type_mapper locally on the filtered AnnData.",
    )
    parser.add_argument(
        "--precomputed-stats",
        default=str(raw_dir / "human_precomputed_stats.h5"),
        help="Path to HMBA Basal Ganglia precomputed_stats.h5 file (used with --run-mapmycells).",
    )
    parser.add_argument(
        "--query-markers",
        default=str(raw_dir / "query_markers.json"),
        help="Path to HMBA Basal Ganglia query_markers.json file (used with --run-mapmycells).",
    )
    parser.add_argument(
        "--regions-level-2",
        nargs="+",
        default=None,
        help="Override target region_level_2 list (default: Substantia nigra and Putamen).",
    )
    parser.add_argument(
        "--restrict-dopa-to-sn",
        action="store_true",
        help="Strictly retain only Substantia nigra nuclei when exporting the dopaminergic dataset.",
    )
    parser.add_argument(
        "--full-output-path",
        default=str(processed_dir / CANONICAL_NAMES["full_dataset"]),
        help="Output path for the full filtered dataset with all HMBA groups.",
    )
    parser.add_argument(
        "--dopaminergic-output-path",
        default=str(processed_dir / CANONICAL_NAMES["dataset_dopaminergic"]),
        help="Output path for the isolated dopaminergic .h5ad dataset passed to Step 02.",
    )
    parser.add_argument(
        "--hmba-extended-json",
        default=None,
        help="Path to save the HMBA MapMyCells extended JSON output.",
    )
    parser.add_argument(
        "--mapmycells-tmp-dir",
        default=None,
        help="Optional custom directory for temporary MapMyCells batch files.",
    )
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--plots-dir", default=str(plots_dir))
    parser.add_argument("--reports-dir", default=str(reports_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step01(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    config_path = Path(args.config)
    config = _resolve_step01_config(config_path)

    if args.chunk_size is not None:
        config["chunk_size"] = max(1000, int(args.chunk_size))
    if args.mapmycells_chunk_size is not None:
        config["mapmycells_chunk_size"] = max(500, int(args.mapmycells_chunk_size))
    if args.mapmycells_batch_size is not None:
        config["mapmycells_batch_size"] = max(10000, int(args.mapmycells_batch_size))
    if args.overwrite_full_dataset:
        config["overwrite_full_dataset"] = True
    if args.use_all_regions:
        config["use_all_regions"] = True
    if args.regions_level_2 is not None:
        config["target_regions_level_2"] = [str(r) for r in args.regions_level_2]
    if args.restrict_dopa_to_sn:
        config["restrict_dopaminergic_to_sn"] = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_h5ad_path = Path(args.raw_h5ad)
    metadata_dir = Path(args.metadata_dir)
    full_out_path = Path(args.full_output_path)

    # 1. Reuse existing full_dataset.h5ad if already streamed, otherwise stream from raw_h5ad_path
    if full_out_path.exists() and not bool(config.get("overwrite_full_dataset", False)):
        LOGGER.info(
            "Found existing filtered dataset at %s; reusing it directly without re-streaming raw H5AD!",
            full_out_path,
        )
        with h5py.File(full_out_path, "r") as f_existing:
            obs_full = read_elem(f_existing["obs"])
        filter_stats = {
            "reused_existing_full_dataset": True,
            "use_all_regions": bool(config["use_all_regions"]),
            "chunk_size": int(config["chunk_size"]),
            "retained_cells_full_dataset": int(len(obs_full)),
            "retained_samples_full_dataset": (
                int(obs_full["ASAP_sample_id"].nunique())
                if "ASAP_sample_id" in obs_full.columns
                else None
            ),
            "retained_subjects_full_dataset": (
                int(obs_full["ASAP_subject_id"].nunique())
                if "ASAP_subject_id" in obs_full.columns
                else None
            ),
            "cells_by_region_level_2": (
                {
                    str(k): int(v)
                    for k, v in obs_full["region_level_2"]
                    .value_counts(dropna=False)
                    .to_dict()
                    .items()
                }
                if "region_level_2" in obs_full.columns
                else {}
            ),
        }
    else:
        if not raw_h5ad_path.exists():
            raise FileNotFoundError(f"Raw unfiltered h5ad file not found: {raw_h5ad_path}")
        LOGGER.info("Building enriched metadata registry from %s...", metadata_dir)
        sample_meta = build_enriched_sample_metadata(metadata_dir)
        obs_full, filter_stats = stream_enrich_and_filter_raw_h5ad(
            raw_h5ad_path=raw_h5ad_path,
            full_out_path=full_out_path,
            sample_meta=sample_meta,
            config=config,
        )

    # 2. Run or apply HMBA Basal Ganglia mapping in checkpointed batches
    hmba_csv_path = Path(args.hmba_mapping_csv) if args.hmba_mapping_csv else None
    precomputed_path = Path(args.precomputed_stats) if args.precomputed_stats else None
    markers_path = Path(args.query_markers) if args.query_markers else None

    obs_full, hmba_stats = run_or_apply_hmba_basal_ganglia_mapping(
        obs_full=obs_full,
        full_out_path=full_out_path,
        config=config,
        hmba_mapping_csv=hmba_csv_path,
        hmba_extended_json=args.hmba_extended_json,
        run_mapmycells=bool(args.run_mapmycells),
        precomputed_stats_path=precomputed_path,
        query_markers_path=markers_path,
        mapmycells_tmp_dir=args.mapmycells_tmp_dir,
    )

    # 3. Stream-extract FILE 2 (Dopaminergic dataset) from full_out_path
    adata_dopa, dopa_stats = isolate_dopaminergic_compartment_from_disk(
        full_out_path=full_out_path,
        obs_full=obs_full,
        config=config,
    )

    dopa_out_path = Path(args.dopaminergic_output_path)
    dopa_out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_dopa.write_h5ad(dopa_out_path)
    LOGGER.info(
        "Saved dopaminergic dataset (%d cells) for Step 02 to: %s",
        adata_dopa.n_obs,
        dopa_out_path,
    )

    # 4. Save QC plots and JSON report
    saved_plots = save_step01_qc_plots(
        obs_full=obs_full,
        adata_dopa=adata_dopa,
        plots_dir=plots_dir,
        config=config,
    )

    report = {
        "config_path": str(config_path),
        "raw_h5ad_path": str(raw_h5ad_path),
        "metadata_dir": str(metadata_dir),
        "full_dataset_h5ad": str(full_out_path),
        "dopaminergic_dataset_h5ad": str(dopa_out_path),
        "parameters": config,
        "regional_filtering": filter_stats,
        "hmba_mapping": hmba_stats,
        "dopaminergic_isolation": dopa_stats,
        "plots": saved_plots,
    }

    report_path = reports_dir / "step01_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    format_pipeline_step01_summary(
        raw_h5ad_path=str(raw_h5ad_path),
        metadata_dir=str(metadata_dir),
        use_all_regions=bool(config["use_all_regions"]),
        chunk_size=int(config["chunk_size"]),
        full_dataset_path=str(full_out_path),
        dopaminergic_dataset_path=str(dopa_out_path),
        report_path=str(report_path),
    )