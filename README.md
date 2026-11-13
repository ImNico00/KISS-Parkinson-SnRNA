# Investigation of Kisspeptins & KISS1R Causal Signaling Pipeline

Repository containing the implementation of an end-to-end Bioinformatics and Systems Biology pipeline for single-nucleus transcriptomic (`snRNA-seq`) analysis of the **ASAP** (*Aligning Science Across Parkinson's*) cohort and causal reconstruction of **KISS1R**-mediated signaling networks in midbrain dopaminergic neurons.

The pipeline automates the entire computational workflow across 6 modular, reproducible steps: out-of-core ingestion of high-dimensional `.h5ad` datasets, taxonomic annotation via **Allen Institute MapMyCells (HMBA Basal Ganglia)**, demographic and transcriptional profiling, pseudobulk differential expression analysis (**EdgeR**), combinatorial Transcription Factor activity estimation (**DoRothEA / Decoupler**), parallel causal network reconstruction (**OmniPath + Corneto / CARNIVAL**), and internal pathway concordance statistical validation on **PROGENy**.

---

## 🚀 System Architecture

The project implements a modular architecture driven by YAML configuration files (`pipeline/config/step01.yml` – `step06.yml`), CLI overrides, and bash wrapper scripts with automatic execution logging (`script/` -> `pipeline/run_logs/`), ensuring scalability on large cellular cohorts and full traceability through structured JSON reports:

- **Out-of-Core Data Engineering & HMBA Mapping (Step 01):**
  - **Metadata Harmonization:** Merges and deduplicates clinical and experimental registries (`SAMPLE.csv`, `SUBJECT.csv`, `CLINPATH.csv`, `ASSAY.csv`) using composite keys and regex-based normalization of sample identifiers (`ASAP_sample_id`).
  - **Streaming & Regional Filtering:** Reads raw `.h5ad` files directly via `h5py` without loading the full expression matrix into RAM, filters target brain regions (*Midbrain / Substantia nigra* and *Striatum / Putamen*) and conditions (`Control`, `PD`), and streams the filtered CSR sparse matrix in `float32` chunks to disk.
  - **HMBA Basal Ganglia Taxonomy:** Integrates Allen Institute `cell_type_mapper` (MapMyCells, taxonomy `CCN20250428`) executed in disk-checkpointed batches (`hmba_shards/`) with automatic temporary directory cleanup to prevent Out-Of-Memory crashes, followed by isolation of midbrain dopaminergic subpopulations (`SN SOX6 Dopa`, `SN-VTR CALB1 Dopa`, `SN-VTR GAD2 Dopa` within subclass `M Dopa`) and UMAP embedding computation.
- **Cohort & Target Receptor Profiling (Step 02):**
  - Deduplicates data at the subject level (`ASAP_subject_id`) for demographic characterization (age, sex, diagnosis, age brackets) and quantifies the percentage of cells expressing the **KISS1R** receptor across clinical subgroups and dopaminergic clusters.
- **Pseudobulk Differential Expression & GSEA Export (Step 03):**
  - Aggregates raw counts into subject-by-cell-type pseudobulk profiles via `decoupler`, applies QC filtering (`min_cells`, `min_counts`), and runs statistical differential expression modeling in **R (`edgeR` via `rpy2`)** through `pertpy`.
  - Dynamically constructs the design formula from non-constant covariates (`~ age [+ sex] + condition_id`) for the **PD vs Control** contrast, generates Volcano plots, and exports pre-ranked `.rnk` files scored by $\text{sign}(\log_2\text{FC}) \times -\log_{10}(p\text{-value})$ ready for **GSEA / WebGestalt**.
- **TF Consensus Optimization & Stratification (Step 04):**
  - Stratifies subjects into `High` vs `Low` groups using two selectable modes (`proxy` to mitigate snRNA-seq sparsity/dropout by combining `KISS1R > 0` with `GNAQ > 0` and at least 3 active proxy markers among `CALB2`, `TH`, `S100B`, `GNRH1`; or `kiss1r` alone).
  - Estimates Transcription Factor activities on **DoRothEA** regulons (confidence levels `A`, `B`, `C`) across 8 scoring algorithms (`ulm`, `waggr`, `viper`, `aucell`, `gsea`, `gsva`, `mlm`, `zscore`) and performs a combinatorial search over all method subsets ($\ge 3$ methods) to identify the consensus maximizing differentially active TFs (Welch's t-test + Benjamini-Hochberg FDR correction).
- **Causal Signaling Network Reconstruction (Step 05):**
  - Builds a signed, directed human Prior Knowledge Network (PKN) from **OmniPath** (`ARN`, `SIGNOR`, `SignaLink3`, `KEGG`, `ACSN`, `Reactome_SignaLink3`, `Reactome_ProtMapper`).
  - Solves MILP/LP optimization problems in parallel (`ThreadPoolExecutor`) via **Corneto (`CarnivalFlow`)** with L1 regularization (`lambda_reg`), simulating activation (`KISS1R = +1`) and inhibition (`KISS1R = -1`) perturbations toward the top differential TFs inferred in Step 04.
- **Internal Pathway & Network Concordance Validation (Step 06):**
  - Statistically validates **PROGENy** pathway activities and CARNIVAL subnetwork topology across PD patients stratified into `KISS1R-High` vs `KISS1R-Low`, combining parametric and non-parametric tests (Welch's t-test, Mann-Whitney U, Cohen's $d$, Hedges' $g$) with continuous rank correlations (Spearman $\rho$, Kendall $\tau$).

---

## 🔬 Scientific Methodology & Statistical Rigor

Transcriptomic analysis of low-abundance GPCRs such as `KISS1R` in post-mortem human brain tissue poses major challenges related to zero-inflation and potential statistical circularity. The pipeline enforces four explicit methodological safeguards:

1. **Adaptive Stratification for Zero-Inflated Distributions:** In Step 06, if the fraction of subjects with zero `KISS1R` expression exceeds the configured threshold (`max_zero_fraction_for_median: 0.50`), the pipeline avoids a degenerate median split by classifying subjects with endogenous detection (`KISS1R > 0`) as `KISS1R-High` versus undetected subjects (`KISS1R = 0`, `KISS1R-Low`), complemented by continuous rank correlations across the entire cohort and an exploratory sub-split within detected subjects (`Detected-High` vs `Detected-Low`).
2. ***A Priori* Pathway Panel Selection & Consensus Inheritance:** The primary **PROGENy** pathway panel is fixed *a priori* strictly based on overlap with active nodes in the CARNIVAL `KISS1R = +1` network, and Benjamini-Hochberg FDR correction is applied exclusively across this panel. Furthermore, Step 06 directly inherits the winning Decoupler consensus method combination selected in Step 04 from `step04_report.json` without re-optimizing on test data.
3. **Circularity Prevention (`KISS1R` Exclusion):** Because `KISS1R` defines the grouping variable (`High` vs `Low`), it is programmatically excluded from `CARNIVAL_Subnetwork` aggregate scores as well as node and edge directional concordance metrics.
4. **Distinction Between mRNA Abundance and Catalytic Flux:** The Step 06 report explicitly separates exploratory mRNA abundance checks on intermediate signaling nodes (kinases and effectors such as `GNAQ`, `HRAS`, `MAPK1`, `ABL1`, `PLK1`, whose activity is post-translationally regulated) from downstream transcriptional footprint scoring (**PROGENy**), clearly framing the evaluation as an *Internal Concordance Assessment*.

---

## 📁 Repository Structure

```text
├── figures/                     # Final publication/presentation figures and visual assets
├── pipeline/
│   ├── config/                  # YAML configuration files for each step (step01.yml .. step06.yml)
│   ├── data/
│   │   ├── raw/                 # Raw .h5ad dataset, ASAP metadata (*.csv), and HMBA MapMyCells lookup files
│   │   └── processed/           # Filtered .h5ad datasets and step output folders (step03_edger .. step06_validation)
│   ├── plots/                   # High-resolution QC and analysis plots (300 DPI)
│   ├── reports/                 # Structured JSON reports with metrics, parameters, and artifact paths (step01_report.json .. step06_report.json)
│   ├── run_logs/                # Execution logs generated by bash wrapper scripts
│   └── src/
│       ├── schemas.py           # Canonical dataset and artifact filenames shared across steps
│       └── utils.py             # I/O utilities, HDF5/AnnData sanitization helpers, and logging setup
├── script/                      # Executable bash scripts to run each pipeline step with automatic logging
├── .env                         # Environment variables / credentials (e.g., ASAP data access)
├── environment.yml              # Conda environment specification (system and R dependencies)
├── poetry.lock                  # Locked Python dependency versions
├── pyproject.toml               # Project configuration and Python dependencies managed via Poetry
└── README.md
```

---

## ⚙️ Environment Setup

The project uses **Poetry** (`package-mode = false`) for strict Python dependency management and requires a working **R** installation to run the `edgeR` statistical model via `rpy2` (also supported via `environment.yml`).

### Prerequisites
- **Python 3.13** (`>=3.13,<3.14`) and **Poetry**
- **R ($\ge 4.2$)** with the Bioconductor `edgeR` package installed (`BiocManager::install("edgeR")`)
- **Allen Institute HMBA Basal Ganglia Taxonomy** reference files (`human_precomputed_stats.h5` and `query_markers.json`) placed inside `pipeline/data/raw/` (required when running MapMyCells locally with `--run-mapmycells`)
- MILP solver for Corneto (`scipy`/HiGHS included by default; optional `gurobi` or `cbc`)

### Installation

1. **Clone the repository.**
2. **Install all Python dependencies (including Allen Institute's `cell-type-mapper`) via Poetry:**
```bash
poetry install
```
*(Optional) If using Conda to manage the Python 3.13 interpreter and R environment with `edgeR`, create the environment from `environment.yml` first:*
```bash
conda env create -f environment.yml
```

---

## 🖥 Running the Pipeline

All steps are designed to be executed via dedicated bash wrapper scripts in `script/`, which forward CLI flags to the Python modules while saving full execution logs in `pipeline/run_logs/`. Default arguments are synchronized with the corresponding YAML configuration files in `pipeline/config/`.

### 1. Out-of-Core Preprocessing and HMBA Mapping (Step 01)
Filters target brain regions, harmonizes ASAP metadata, runs HMBA cell-type classification in disk-checkpointed batches, and exports the isolated dopaminergic dataset:
```bash
./script/run_step01_with_logs.sh --run-mapmycells --chunk-size 50000 --mapmycells-batch-size 200000
```
*Key flags:*
- `--run-mapmycells` / `--hmba-mapping-csv <path>`: Run MapMyCells locally or import a precomputed CSV annotation file.
- `--use-all-regions` / `--regions-level-2`: Retain all brain regions or override target regional filtering (default: *Substantia nigra* and *Putamen* using UBERON ontology).
- `--restrict-dopa-to-sn`: Restrict dopaminergic neuron extraction strictly to *Substantia nigra* nuclei.
- `--overwrite-full-dataset`: Force re-streaming of `full_dataset.h5ad` even if it already exists on disk.

### 2. Demographic Profiling and Baseline KISS1R Expression (Step 02)
Generates cohort demographic summaries and exploratory plots (UMAP of dopaminergic subclusters, global and sex/group-stratified `KISS1R` dotplots, age and sex distribution barplots):
```bash
./script/run_step02_with_logs.sh
```

### 3. Pseudobulk Differential Expression Analysis with EdgeR (Step 03)
Aggregates single-nucleus counts by subject and neuronal subtype and tests **PD vs Control** while adjusting for age and sex covariates:
```bash
./script/run_step03_with_logs.sh --sex-group both --min-cells 20 --min-counts 1000 --min-subjects-group 5
```
*Key flags:*
- `--sex-group {male,female,both}`: Select target sex cohorts for differential expression analysis.
- `--combined`: When combined with `--sex-group both`, pools male and female subjects into a single pseudobulk (`ALL`) rather than analyzing them separately.

### 4. Transcription Factor Consensus Optimization (Step 04)
Stratifies subjects into `High` vs `Low` groups and searches for the optimal consensus combination of `decoupler` methods on **DoRothEA** regulons to export TF measurement vectors for CARNIVAL:
```bash
./script/run_step04_with_logs.sh --cohorts pd_male pd_female --stratification proxy --min-cells 20
```
*Key flags:*
- `--cohorts {control_male,control_female,pd_male,pd_female,male,female,all}`: Select one or more target cohorts.
- `--stratification {proxy,kiss1r}`: Stratification strategy for `High` vs `Low` (`proxy` uses `KISS1R` or `GNAQ` + proxy markers; `kiss1r` uses `KISS1R > 0` only).

### 5. Causal Signaling Network Reconstruction (Step 05)
Automatically discovers Step 04 TF CSV outputs, retrieves the signed PKN from **OmniPath**, and solves **CARNIVAL** optimizations in parallel across target perturbation states:
```bash
./script/run_step05_with_logs.sh --perturbation-node KISS1R --target-states 1 -1 --top-tfs 50 --solver scipy --threads 3
```
*Key flags:*
- `--perturbation-node`: Input perturbation gene symbol (default: `KISS1R`).
- `--target-states`: Perturbation states to simulate (default: `1 -1`).
- `--top-tfs`: Maximum number of top differential TFs included as outputs (default: `50`).
- `--lambda-reg`: L1 regularization penalty for `CarnivalFlow` (default: `0.001`).
- `--solver {scipy,gurobi,highs,cbc}` and `--threads`: Optimization solver backend and parallel worker thread count.

### 6. Internal Validation on PROGENy Pathways and CARNIVAL Topology (Step 06)
Performs statistical and transcriptomic concordance validation comparing `KISS1R-High` vs `KISS1R-Low` PD patients across *a priori* selected **PROGENy** pathways and evaluates directional concordance on reconstructed CARNIVAL nodes and edges:
```bash
./script/run_step06_with_logs.sh
```

---

## 📊 Outputs and Generated Artifacts

Upon completion, the pipeline produces:

1. **Processed Datasets (`pipeline/data/processed/`):**
   - `full_dataset.h5ad` and `dataset_dopaminergic.h5ad` annotated with HMBA taxonomy and harmonized clinical metadata.
   - `step03_edger/`: Full EdgeR differential expression tables (`edgeR_<COHORT>_PDvsControl_<CELLTYPE>.csv`) and pre-ranked GSEA files (`webgestalt_data_<COHORT>_<CELLTYPE>.rnk`).
   - `step04_carnival/`: Differential TF measurement tables (`TF_Measurements_CARNIVAL_<COHORT>_<MODE>.csv`).
   - `step05_networks/`: Active causal subnetworks (`Carnival_Edges_*.csv` and `Carnival_Nodes_*.csv`) ready for direct import into **Cytoscape**.
   - `step06_validation/`: 7 CSV tables per cohort containing subject stratification, pathway activity scores, differential statistics (`Welch` + `Mann-Whitney U` + `Hedges' g`), continuous rank correlations (`Spearman` + `Kendall`), and exploratory node/edge validation metrics.
2. **Visualizations (`pipeline/plots/`):**
   - Regional cohort QC barplots, dopaminergic UMAPs, `KISS1R` expression dotplots, pseudobulk QC plots, Volcano plots and DEG summary barplots, Z-score pathway heatmaps ordered by `KISS1R`, $\Delta\text{Activity}$ forest plots with 95% confidence intervals, and continuous regression scatterplots.
3. **Traceability Reports & Logs (`pipeline/reports/` & `pipeline/run_logs/`):**
   - `step01_report.json` – `step06_report.json` recording resolved parameters, filtering counts, statistical test outcomes, and output artifact paths alongside complete execution logs.