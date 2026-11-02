from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import warnings
import pandas as pd

import scipy.sparse as sp
import numpy as np
import scanpy as sc
import squidpy as sq
import anndata as ad

from pipeline.src.schemas import CANONICAL_NAMES
from pipeline.src.utils import configure_logging

LOGGER = logging.getLogger(__name__)

def generate_gsea_rnk(adata: Any, name: str, out_dir: Path) -> None:
    """
    Calcola l'espressione media assoluta dei geni, li ordina in modo decrescente
    e salva un file .rnk formattato per GSEA.
    """
    # 1. Calcolo espressione media
    if sp.issparse(adata.X):
        mean_expression = np.asarray(adata.X.mean(axis=0)).flatten()
    else:
        mean_expression = np.asarray(adata.X).mean(axis=0).flatten()

    # 2. Creazione DataFrame (i geni vengono messi in uppercase per GSEA/MSigDB)
    df_rnk = pd.DataFrame({
        'Gene': adata.var_names.str.upper(), 
        'Score': mean_expression
    })

    # 3. Ordinamento
    df_rnk = df_rnk.sort_values(by='Score', ascending=False)

    # 4. Salvataggio
    rnk_filename = out_dir / f"{name}_expression.rnk"
    df_rnk.to_csv(rnk_filename, sep='\t', header=False, index=False)
    
    LOGGER.info(f"File GSEA .rnk salvato: {rnk_filename} ({len(df_rnk)} geni validi).")

def get_h5_files_filtered_from_dir(dataset_dir: Path) -> dict[str, Any]:
    """
    Restituisce una lista di file .h5 filtrati dalla directory specificata.
    """
    h5_files = list(dataset_dir.glob("*_filtered_feature_bc_matrix.h5"))
    filtered_files = [file for file in h5_files if file.is_file()]

    adata_list = {}
    for file in filtered_files:
        name_file = file.name.split("_")[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adata_list[name_file] = sc.read_10x_h5(file)
            adata_list[name_file].var_names_make_unique()
    return adata_list

def get_csv_files_positions_from_dir(dataset_dir: Path, adata_list: dict[str, Any]) -> dict[str, Any]:
    """
    Restituisce una lista di file .csv contenenti le posizioni spaziali dalla directory specificata.
    """
    csv_files = list(dataset_dir.glob("*_tissue_positions_list.csv.gz"))
    filtered_files = [file for file in csv_files if file.is_file()]

    for file in filtered_files:
        name_file = file.name.split("_")[0]
        posizioni = pd.read_csv(file, header=None, index_col=0)
        posizioni.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        adata_list[name_file].obsm['spatial'] = posizioni.loc[adata_list[name_file].obs_names, ['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
        LOGGER.info(f"Grafo {name_file} pronto per l'estrazione: {adata_list[name_file].shape[0]} nodi (spot) e {adata_list[name_file].shape[1]} feature (geni)")
    return adata_list

def format_pipeline_step01_summary(
    *,
    dataset_dir: str,
    output_dir: str
) -> str:
    summary = (
        "Pipeline Step01 summary:\n"
        f"- Original Dataset Directory: {dataset_dir}\n"
        f"- Output Directory: {output_dir}"
    )
    return summary

def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step01 Data Preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    raw_dir = default_repo_root / "pipeline" / "data" / "raw"
    parser.add_argument("--dataset-dir", default=str(raw_dir))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser

def run_step01(args: argparse.Namespace) -> None:
    configure_logging(args.logging_level)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    # 1. Carichiamo la matrice di espressione genica
    h5_dataset = get_h5_files_filtered_from_dir(dataset_dir)
    if not h5_dataset:
        LOGGER.error("No filtered .h5 files found in the dataset directory.")
        return
    adata_list = get_csv_files_positions_from_dir(dataset_dir, h5_dataset)

    LOGGER.info(f"Loaded {len(adata_list)} datasets from {dataset_dir}")

    genes_to_keep = [
        'KISS1', 
        'KISS1R', 
        'GNAQ', 
        'PIK3CA', 
        'AKT1', 
        'GSK3B', 
        'SNCA', 
        'TH', 
        'BDNF', 
        'NTRK2', 
        'NPVF', 
        'GNRHR', 
        'GNRH1', 
        'TAC3',
        'TAC2',
        'TACR3', 
        'PLCB1', 
        'GNA11', 
        'ESR2',
        'PPP5C',
        'NCOA5',
        'MAPK1',
        'MAPK3',
        'CREB1',
        'HMOX1',
        'CALB1',
        'ARRB2',
        'PRKCA',
        'ITPR1',
        'DUSP4',
        'PTEN',
        'CASP3',
        'CASP9',
        'CYCS',
        'APAF1',
        'PRKN',
        'PINK1',
        'PARK7',
        'BCL2',
        'BAX',
        'BAD',
        'CACNA1D',
        'PLCB2',
        'PLCB3'
    ]

    genomes_dict = {}
    for name, adata in adata_list.items():
        genome = adata.var['genome'].unique().tolist()[0]
        if genome not in genomes_dict:
            genomes_dict[genome] = []
        genomes_dict[genome].append(adata)
    
    # 2. Per ogni specie uniamo i dati, normalizziamo e creiamo il .rnk
    for genome, adatas in genomes_dict.items():
        LOGGER.info(f"Elaborazione file .rnk per il genoma '{genome}' ({len(adatas)} campioni)...")
        
        # Concateniamo i raw data della stessa specie
        adata_species = ad.concat(adatas, join="inner")
        
        # Applichiamo QC di base e normalizzazione (stessa logica del post-processing)
        sc.pp.filter_cells(adata_species, min_counts=500)
        sc.pp.normalize_total(adata_species, target_sum=1e4)
        sc.pp.log1p(adata_species)
        
        # Generiamo il file .rnk per questo genoma
        generate_gsea_rnk(adata_species, f"all_samples_grouped_{genome}", out_dir)
        
        # Liberiamo la RAM
        del adata_species

    # =====================================================================
    # POST-PROCESSING INDIVIDUALE DEI CAMPIONI
    # =====================================================================
    for name, adata in adata_list.items():
        genome = adata.var['genome'].unique().tolist()[0]
        adata_list[name] = run_step01_postprocessing(adata, name, genes_to_keep, genome)

    LOGGER.info(
        "\n%s",
        format_pipeline_step01_summary(
            dataset_dir=str(dataset_dir),
            output_dir=str(out_dir)
        ),
    )

def run_step01_postprocessing(
    adata,
    name: str,
    genes_list: list[str],
    genome
) -> Any:

    genes_to_keep = genes_list.copy()
    gene_to_remove = 'TAC3' if 'mm10' in genome else 'TAC2'

    if gene_to_remove in genes_to_keep:
        genes_to_keep.remove(gene_to_remove)

    gene_lookup = {str(gene).lower(): gene for gene in adata.var_names}

    # 1. Quality Control
    sc.pp.filter_cells(adata, min_counts=500)

    target_mask = (
        adata.var_names.astype(str)
        .str.lower()
        .isin({g.lower() for g in genes_to_keep})
    )

    gene_counts = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    keep_genes = (gene_counts >= 10) | target_mask
    adata = adata[:, keep_genes].copy()

    # 2. Normalizzazione e log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 3. Highly Variable Genes
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=3000)

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("Scanpy non ha creato la colonna 'highly_variable'")

    # 4. Forziamo i geni target ad essere inclusi
    for gene in genes_to_keep:
        var_name = gene_lookup.get(gene.lower())
        if var_name is not None:
            adata.var.loc[var_name, "highly_variable"] = True

    # 5. Manteniamo solo gli HVG + geni target
    mask = adata.var["highly_variable"].fillna(False).to_numpy(dtype=bool)
    adata = adata[:, mask].copy()

    # 6. Grafo spaziale
    sq.gr.spatial_neighbors_knn(adata, n_neighs=6)

    return adata