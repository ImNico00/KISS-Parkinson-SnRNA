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

# Assicurati di avere queste importazioni in cima al file!
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

LOGGER = logging.getLogger(__name__)

def generate_gsea_rnk_pseudobulk(adata: sc.AnnData, out_dir: Path, target_segment: str = 'TH') -> None:
    LOGGER.info(f"--- Inizio analisi PSEUDOBULK per il segmento {target_segment} ---")

    # MODIFICA: Usiamo 'segment' al posto di 'cell_type'
    if 'segment' not in adata.obs.columns:
        LOGGER.error("Colonna 'segment' non trovata nei metadati!")
        return

    mask_ct = adata.obs['segment'] == target_segment
    adata_sub = adata[mask_ct].copy()
    
    if adata_sub.n_obs == 0:
        LOGGER.error("Nessuna ROI trovata per l'analisi.")
        return
    
    LOGGER.info(f"ROI totali trovate: {adata_sub.n_obs}. Aggregazione per paziente in corso...")

    var_names = adata_sub.raw.var_names if adata_sub.raw is not None else adata_sub.var_names

    # AGGREGAZIONE MEMORY-SAFE
    donor_counts = {}
    # MODIFICA: 'donor_id' diventa 'asap_subject_id'
    for donor in adata_sub.obs['asap_subject_id'].dropna().unique():
        mask_donor = (adata_sub.obs['asap_subject_id'] == donor).values
        
        X_donor = adata_sub.raw.X[mask_donor] if adata_sub.raw is not None else adata_sub.X[mask_donor]
        donor_sum = X_donor.sum(axis=0)
        
        if hasattr(donor_sum, "A1"): 
            donor_counts[donor] = donor_sum.A1
        else:
            donor_counts[donor] = np.squeeze(np.asarray(donor_sum))

    pseudobulk_counts = pd.DataFrame.from_dict(donor_counts, orient='index', columns=var_names)
    pseudobulk_counts = pseudobulk_counts.round().astype(int)
    pseudobulk_counts = pseudobulk_counts.loc[:, pseudobulk_counts.sum(axis=0) > 0]
    
    # Creazione metadati
    meta = (
        adata_sub.obs[
            # MODIFICA: Aggiornati i nomi delle variabili cliniche
            ["asap_subject_id", "gp2_phenotype", "sex"]
        ]
        .drop_duplicates("asap_subject_id")
        .set_index("asap_subject_id")
    )
    # Assicuriamoci che gli indici combacino e rimuoviamo eventuali NaN
    meta = meta.loc[pseudobulk_counts.index].dropna(subset=['gp2_phenotype'])
    pseudobulk_counts = pseudobulk_counts.loc[meta.index]

    LOGGER.info(f"Pazienti totali (per stima dispersione globale): {len(meta)}")
    LOGGER.info(f"Distribuzione Diagnosi: \n{meta['gp2_phenotype'].value_counts()}")

    LOGGER.info("Avvio calcoli PyDESeq2...")
    
    inference = DefaultInference(n_cpus=4)
    
    dds = DeseqDataSet(
        counts=pseudobulk_counts, 
        metadata=meta, 
        design_factors="sex + gp2_phenotype", # Corretto con la variabile clinica giusta
        quiet=True,
        inference=inference
    )
    dds.deseq2()
    
    # MODIFICA: Contrasto impostato su gp2_phenotype (Assumiamo PD vs Control, adatta se i nomi differiscono nel CSV)
    # Potrebbe essere necessario cambiare "PD" con "Parkinson's Disease" in base a cosa c'è nel tuo CSV
    stat_res = DeseqStats(dds, contrast=["gp2_phenotype", "PD", "Control"], quiet=True, inference=inference)
    stat_res.summary()
    res_df = stat_res.results_df.dropna(subset=['stat'])
    
    df_rnk = res_df[['stat']].reset_index()
    df_rnk.columns = ['Gene', 'Score']
    df_rnk['Gene'] = df_rnk['Gene'].astype(str).str.upper()
    df_rnk = df_rnk.sort_values('Score', ascending=False)
    
    file_name = target_segment.replace(" ", "_").lower()
    rnk_path = out_dir / f'GSEA_Pseudobulk_{file_name}.rnk'
    df_rnk.to_csv(rnk_path, sep='\t', header=False, index=False)
    
    LOGGER.info(f"File GSEA generato con successo: {rnk_path}")


def get_dataset_from_h5ad(file_path: Path) -> ad.AnnData:
    adata : ad.AnnData = sc.read_h5ad(file_path)

    if 'feature_name' in adata.var.columns:
        adata.var_names = adata.var['feature_name'].astype(str)
        adata.var_names_make_unique()
    elif 'gene_symbols' in adata.var.columns:
        adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names_make_unique()

    # Mantenuto il tuo check anatomico, con fallback sicuro
    if 'region_level_2' in adata.obs.columns:
        mask_nigra = adata.obs['region_level_2'].str.contains('substantia nigra', case=False, na=False)
        adata = adata[mask_nigra].copy()
    else:
        LOGGER.warning("Filtro anatomico ignorato temporaneamente (i metadati clinici verranno uniti allo step successivo).")

    return adata


def format_pipeline_step01_summary(*, dataset_dir: str, output_dir: str) -> str:
    return f"Pipeline Step01 summary:\n- Original Dataset Directory: {dataset_dir}\n- Output Directory: {output_dir}"


def build_step01_parser(default_repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step01 Data Preparation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    processed_dir = default_repo_root / "pipeline" / "data" / "processed"
    raw_dir = default_repo_root / "pipeline" / "data" / "raw"
    parser.add_argument("--metadata-path", default=str(raw_dir / "parkinsonsroadmap/merged_data.csv"))
    parser.add_argument("--dataset-dir", default=str(raw_dir / "edwards-pmdbs-spatial-geomx-th/spatial_geomx/"))
    parser.add_argument("--output-dir", default=str(processed_dir))
    parser.add_argument("--logging-level", default="INFO")
    return parser


def run_step01(args: argparse.Namespace) -> None:
    # configure_logging(args.logging_level)  # Scommenta se definito
    # plt.rcParams.update({'figure.autolayout': True})
    sc.set_figure_params(dpi=300, facecolor='white')
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    # 1. Percorsi
    percorso_final = "/mnt/h/asap-cohort.final.h5ad"
    percorso_unfiltered = "/mnt/h/asap-cohort.merged_cleaned_unfiltered.h5ad"
    percorso_csv = "/mnt/h/MOUSE.csv"
    percorso_output = "/mnt/h/dataset_kisspeptina_leggero.h5ad"

    LOGGER.info("1. Recupero le annotazioni cellulari (phenotype) dal file final...")
    adata_final = sc.read_h5ad(percorso_final, backed='r')
    metadati_base = adata_final.obs.copy()
    
    # Liberiamo subito la memoria del file final che non ci serve più
    del adata_final
    gc.collect()

    LOGGER.info("2. Caricamento del dataset in modalità backed (risparmio RAM)...")
    adata_unfiltered = sc.read_h5ad(percorso_unfiltered, backed='r')

    LOGGER.info("3. Filtraggio e caricamento a CHUNK per evitare crash di RAM...")
    cellule_valide = metadati_base.index.intersection(adata_unfiltered.obs_names)
    
    # IMPORTANTE: h5py è molto più efficiente se legge i dati in ordine sequenziale.
    # Filtriamo mantenendo l'ordine esatto in cui le cellule appaiono nel file gigante.
    valid_mask = adata_unfiltered.obs_names.isin(cellule_valide)
    cellule_valide_ordinate = adata_unfiltered.obs_names[valid_mask]

    chunk_size = 10000  # Se crasha ancora, abbassa a 10000. Se va liscio e vuoi velocizzare, prova 50000.
    adatas = []

    for i in range(0, len(cellule_valide_ordinate), chunk_size):
        chunk_cells = cellule_valide_ordinate[i:i + chunk_size]
        LOGGER.info(f"   - Elaborazione chunk: cellule da {i} a {i + len(chunk_cells)} di {len(cellule_valide_ordinate)}...")
        
        # Estrae solo questo piccolo blocco e lo mette in RAM
        adata_chunk = adata_unfiltered[chunk_cells, :].to_memory()
        
        # Assegna i metadati corretti solo per questo chunk
        adata_chunk.obs = metadati_base.loc[chunk_cells].copy()
        
        adatas.append(adata_chunk)

    LOGGER.info("4. Concatenazione dei blocchi...")
    # Uniamo tutti i chunk in un unico oggetto AnnData in RAM
    adata = sc.concat(adatas, merge="same")
    
    # Pulizia aggressiva della memoria prima dei calcoli successivi
    del adatas
    del adata_unfiltered
    gc.collect()

    LOGGER.info("5. Caricamento dei metadati biologici (topi)...")
    df_mouse = pd.read_csv(percorso_csv)

    LOGGER.info("6. Integrazione Sesso e Modello Malattia...")
    adata.obs['ASAP_mouse_id'] = adata.obs['sample'].astype(str).str.extract(r'(ASAP_MOUSE_\d+)')

    mapping_sex = dict(zip(df_mouse['ASAP_mouse_id'], df_mouse['sex']))
    mapping_strain = dict(zip(df_mouse['ASAP_mouse_id'], df_mouse['strain']))

    adata.obs['sex'] = adata.obs['ASAP_mouse_id'].map(mapping_sex)
    adata.obs['strain'] = adata.obs['ASAP_mouse_id'].map(mapping_strain)

    adata.obs['Gruppo_Sperimentale'] = np.where(
        adata.obs['strain'].astype(str).str.contains('MitoPark', case=False, na=False), 
        'Parkinson (MitoPark)', 
        'Control (WT)'
    )

    LOGGER.info("\nRiepilogo delle cellule per gruppo sperimentale:")
    LOGGER.info(adata.obs['Gruppo_Sperimentale'].value_counts())

    LOGGER.info("\nRiepilogo delle cellule per sesso:")
    LOGGER.info(adata.obs['sex'].value_counts())

    LOGGER.info(f"\n7. Salvataggio del NUOVO SUPER-FILE in: {percorso_output}")
    adata.write_h5ad(percorso_output)

    LOGGER.info("\n✅ Processo completato! Il file è pronto con TUTTI i geni e TUTTI i metadati.")

    #LOGGER.info("\n%s", format_pipeline_step01_summary(dataset_dir=str(dataset_dir), output_dir=str(out_dir)))