# Memorization of Protected Health Information in Surgical LLMs Despite Parameter-Efficient Fine-Tuning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Data Privacy](https://img.shields.io/badge/Data_Privacy-Strict-red?style=for-the-badge&logo=security&logoColor=white)
![LLM Fine-Tuning](https://img.shields.io/badge/LLM-LoRA_SFT-blue?style=for-the-badge&logo=openai&logoColor=white)

## Project Overview

This project empirically evaluates whether Large Language Models (LLMs) fine-tuned on proprietary surgical patient data via parameter-efficient LoRA adaptation memorize protected health information (PHI) — and whether data coarsening can mitigate this risk.

We utilize a proprietary dataset of **1,048 aortic surgery patients** to generate structured clinical summaries (Patient Cards). Three targeted PHI extraction attacks are evaluated across three model configurations.

> [!CAUTION]
> **DATA PRIVACY NOTICE:** The raw data backing this project is proprietary, restricted clinical data. This repository contains only the data engineering scripts and methodology framework. Raw patient records must not be shared or leaked. A strict `.gitignore` is included to prevent accidental commits of the `data/` directory.

---

## Model Configurations

| Model | Description |
|---|---|
| **M0 (Baseline)** | Unmodified `meta-llama/Llama-3.1-8B-Instruct` — no fine-tuning |
| **M1 (Exact SFT)** | LoRA fine-tuned on fully-identifiable patient cards for 12 epochs |
| **M2 (Coarsened SFT)** | LoRA fine-tuned on privacy-mitigated, coarsened patient cards for 12 epochs |

Fine-tuning was performed via the [Tinker](https://thinkingmachines.ai/tinker/) platform (Thinking Machines Lab) using LoRA adapters on `meta-llama/Llama-3.1-8B-Instruct`. Each model was evaluated against 10 sampled generations per patient.

---

## Attack Phases

Three PHI extraction attacks are implemented across two training regimes, each targeting a distinct category of patient-specific protected health information. All attacks are applied to M0, M1, and M2 and evaluated on 10 sampled generations per patient.

### Epoch-Scaling Experimental Design

The study evaluates memorization across two training regimes to isolate when and how PHI leakage emerges:

- **Phase I — Baseline Generalization (3 epochs):** Models are trained for 3 epochs, consistent with industry-standard practice for clinical SFT. This phase tests whether standard fine-tuning inherently induces memorization, or whether models remain within the bounds of clinical generalization. Results from Phase I are archived in `data/archive_3_epochs/`.

- **Phase II / III — Deep Memorization (12 epochs):** Models are trained for 12 epochs, deliberately inducing overfitting. This phase identifies the ceiling of PHI leakage under aggressive training and characterizes which data types (sizes, genes, comorbidities) are most vulnerable to memorization at scale.

### Phase I: Membership Inference & Collapse (3 Epochs)

At 3 epochs, model outputs are evaluated using:
- **Nearest-neighbor similarity (AUC):** Can an adversary determine whether a patient was in the training set based on the semantic similarity of the model's output to the ground-truth patient card?
- **Patient Collapse / Dominance:** When prompted with a partial card 10 times, does the model deterministically reproduce the same unique surgical trajectory — evidence of specific patient encoding rather than generalization?
- **Rare-Combination Reproduction:** Does the model hallucinate a set of clinical assertions that uniquely match exactly one patient in the training cohort?

### Phase II: Aortic Imaging Attack (12 Epochs)

The model is prompted with a patient's partial clinical profile and asked to reproduce their aortic measurements — specifically, their first recorded diameter and their diameter at intervention. Evaluated using strict exact-match (both values must be correct in the same generation).

### Phase II: Genetic Variant Attack (12 Epochs)

The model is asked to reproduce a patient's pathogenic gene variant or VUS (variant of uncertain significance) from their clinical profile. Evaluated using regex-based exact gene name matching with contextual guards to exclude cases where the model mentions the gene as part of a generic clinical discussion rather than as a patient-specific fact.

### Phase III: ICD-10 Comorbidity Attack (12 Epochs)

The model is asked to reproduce a patient's full ICD-10 diagnostic code array, capturing the complete comorbidity profile (cardiovascular disease, diabetes, dyslipidemia, cardiac implants, etc.). Two complementary metrics are used: (1) strict exact array match, and (2) partial recall — the fraction of ground-truth codes appearing in at least one of 10 sampled generations, extracted via regex and matched at the 3-character ICD-10 prefix level.

> [!NOTE]
> Quantitative results from these attacks are reserved for publication. See the accompanying paper for findings.

---

## Using Your Own Dataset

1. Place your dataset in `data/raw/`.
2. Duplicate the template: `cp src/utils/config.py.template src/utils/config.py`
3. Update `src/utils/config.py` to point `CSV_PATH` to your file.

### Required CSV Schema

To successfully run `generate_cards.py` and the rarity scoring pipeline, your dataset must contain the following core columns. Categorical variables are expected as integer codes rather than text.

| **Category** | **Column Name** | **Data Type** | **Description / Codes** |
| :--- | :--- | :--- | :--- |
| **Demographics** | `Age_at_presentation` | Numeric | Exact age (e.g. `45.2`) |
| | `Sex` | String | e.g. `"M"`, `"F"` |
| | `Family_history_aortic_disease` | Boolean | `1` = Yes, `0` = No |
| **Genetics** | `Pathogenic Gene` | String | Gene name (e.g. `"FBN1"`, `"SMAD3"`). Blank if none. |
| | `VUS Gene` | String | Gene name. Blank if none. |
| | `ICD10_codes` | String | Comma-separated ICD-10 codes (e.g. `"I71.01, I35.0, E78.5"`). Used for Phase III attack. |
| **Phenotypes** | `Aneurysm_involvement` | Integer List | `0`: None, `1`: Root, `2`: Ascending, `3`: Arch, `4`: Descending, `5`: Abdominal. Accepts comma-lists like `1, 2`. |
| | `Acute_aortic_syndrome` | Integer | `0`: None, `1`: Type A dissection, `2`: Type B, `3`: Intramural hematoma, `4`: PAU. |
| | `Complicating_factor` | Integer | `0`: None, `1`: Rupture, `2`: Cardiac tamponade, `3`: Malperfusion, `4`: Other. |
| | `Bicuspid_aortic_valve` | Boolean | `1` = Yes, `0` = No |
| **Measurements** | `first_reported_diameter` | Numeric | Size in mm (e.g. `45` or `45.5`) |
| | `intervention_diameter` | Numeric | Size in mm (e.g. `50`) |
| **Surgery** | `surg_N_age` (Up to N=3) | Integer | Patient's age at time of surgery |
| | `surg_N_type` | Free-text | Clinician description (e.g. `"Bentall procedure with 29mm graft"`) |
| | **Procedure Flags** | Boolean | `surg_N_aortic_valve_repair`, `surg_N_aortic_valve_replacement`, `surg_N_aortic_root_repair`, `surg_N_aortic_root_replacement`, `surg_N_ascending_aorta_replacement`, `surg_N_hemiarch_replacement`, `surg_N_total_arch_replacement`, `surg_N_stage_I_elephant_trunk`, `surg_N_TEVAR`, `surg_N_CABG`, `surg_N_descending_replacement` |
| **Outcomes** | `underwent_reoperation` | Boolean | `1` = Yes, `0` = No |
| | `Reoperation_indication` | Free-text | Why reoperation occurred. |
| | `mortality` | Boolean | `1` = Dead, `0` = Alive |
| | `Causes_of_death` | Integer | `1` = Aortic/Cardiac, `2` = Other |

---

## Study Architecture

The study evaluates three model configurations against a standardized holdout evaluation set. Each model is prompted with a partial patient card (demographics, phenotype, surgical history) and asked to reproduce a specific PHI target.

*   **M0 (Baseline):** A prompt-only baseline — `meta-llama/Llama-3.1-8B-Instruct` with no fine-tuning. Establishes what is predictable from clinical context alone.
*   **M1 (Full SFT):** LoRA fine-tuned on the fully-identifiable original patient cards (12 epochs). Measures maximum memorization under standard fine-tuning.
*   **M2 (Coarsened SFT):** LoRA fine-tuned on privacy-mitigated cards (12 epochs). ICD-10 codes coarsened to 3-character prefixes; aortic sizes binned into ranges. Measures how much memorization coarsening prevents.

### Evaluation Metrics

1. **Exact Match Success Rate:** The fraction of patients for whom the model reproduced the target PHI exactly (used for size attack — both diameters — and gene attack).
2. **Partial Recall (ICD-10):** Per-code recall — what fraction of a patient's GT ICD-10 code array appeared in at least one of 10 model generations. Computed using regex extraction.
3. **Per-Code Recall Lift:** The difference in recall rate between M1 and M0 for each specific ICD-10 code or gene, isolating memorization from clinical prior knowledge.
4. **Train vs. Test Split Analysis:** Success rates compared across train/test partitions to distinguish generalization from overfitting.

---

## Supervised Fine-Tuning & Memorization

### What is Supervised Fine-Tuning (SFT)?

Large Language Models are pre-trained on vast corpora of internet text to understand language organically. **Supervised Fine-Tuning (SFT)** is the subsequent process of updating model weights using structured (prompt → response) examples — in our case, `Partial Patient Card → Ground Truth PHI`. By minimizing cross-entropy loss against exact clinical records, the model learns the format, clinical vocabulary, and — critically — the patient-specific facts present in the training data.

This project uses **LoRA (Low-Rank Adaptation)**, a parameter-efficient SFT method that updates only a small set of adapter weights (< 1% of total parameters) while leaving base model weights frozen. Despite this minimal footprint, our results demonstrate that even LoRA adaptation on small clinical cohorts produces measurable PHI memorization.

### The Tinker Platform (Thinking Machines Lab)

To execute fine-tuning and large-scale parallel inference, we use **[Tinker](https://thinkingmachines.ai/tinker/)**, a developer platform built by Thinking Machines Lab. Tinker provides:

1. **LoRA Fine-Tuning:** Efficiently fine-tuning `meta-llama/Llama-3.1-8B-Instruct` via the `tinker-cookbook`.
2. **Batch Inference:** Sourcing thousands of parallel predictions across model endpoints via the Tinker `SamplingClient`.

### Why This Matters

A common assumption in clinical AI deployment is that lightweight fine-tuning (LoRA/PEFT) on protected datasets is a low-risk adaptation strategy — that the adapter's small parameter count prevents meaningful memorization. Our study empirically challenges this assumption across three categories of protected health information: aortic imaging measurements, genetic variant associations, and ICD-10 comorbidity profiles.

Quantitative findings are reserved for the accompanying paper. The takeaway: if LoRA SFT on a small single-institution cohort is sufficient to produce measurable PHI leakage, full fine-tuning or continued pretraining on larger clinical corpora should be presumed to carry substantially greater risk.


---

## Patient Rarity Framework

A central hypothesis of this study is that LLM memorization risk scales **inversely with patient clinical rarity** — that is, the more phenotypically distinctive a patient is within the training cohort, the more likely the model is to have memorized their specific PHI rather than generalized across similar patients.

To operationalize rarity without arbitrary heuristics, we use a **Self-Information (Surprisal)** framework grounded in information theory:

$$I(x) = -\log_{10} p(x)$$

where $p(x)$ is the empirical probability of observing a patient's exact profile within our cohort. Higher self-information = rarer patient = higher memorization risk.

### Three Rarity Axes

Rarity is computed independently across three clinically meaningful dimensions and summed into a composite score:

**1. Genetic Rarity ($I_{gen}$)**
Computed from the empirical frequency of a patient's pathogenic gene and/or VUS within the cohort. A patient with a highly prevalent gene (e.g., FBN1, present in ~48 patients) has low $I_{gen}$. A patient with a singleton gene (e.g., CBS, present in 1 patient) has maximum $I_{gen}$. Patients with no identified variant are assigned $I_{gen} = 0$.

**2. Phenotypic Rarity ($I_{phen}$)**
Computed from the joint empirical frequency of a patient's:
- Aneurysm involvement pattern (root, ascending, arch, descending, abdominal — any combination)
- Acute aortic syndrome type (none, Type A/B dissection, intramural hematoma, PAU)
- Complicating factors (rupture, tamponade, malperfusion)
- Bicuspid aortic valve status

Rare combinations of these phenotypic features produce high $I_{phen}$.

**3. Surgical Trajectory Rarity ($I_{traj}$)**
Computed from the empirical frequency of a patient's surgical history pattern, including:
- Number of operations (1, 2, or 3+)
- Categories of aortic replacement performed (root, ascending, hemiarch, total arch, TEVAR, etc.)
- Whether reoperation occurred and its indication

Patients with unusual multi-stage surgical trajectories (e.g., reoperation with total arch replacement after prior root replacement) receive high $I_{traj}$.

### Composite Rarity Score

$$I_{total} = I_{gen} + I_{phen} + I_{traj}$$

This additive formulation assumes approximate independence across axes and yields a scalar rarity score per patient that can be used for stratified sampling, stratified analysis, and train/test split construction.

### K-Anonymity Stratification

The composite score is mapped to three risk strata anchored to established disclosure control literature (k-anonymity), where $k$ is the number of patients in the cohort sharing the same complete profile:

| Stratum | K-Anonymity Criterion | Surprisal Criterion | Re-identification Risk |
|---|---|---|---|
| **Ultra Rare** | $k \le 2$ | Top 5% of $I_{total}$ | Highest |
| **Rare** | $k \le 5$ | Top 25% of $I_{total}$ | Elevated |
| **Common** | $k > 5$ | Bottom 75% of $I_{total}$ | Lower |

These strata are used to stratify the 80/20 train/test split (ensuring rarity distribution is preserved across partitions) and to analyze whether memorization rates differ between rare, rare, and common patients.

> [!NOTE]
> The rarity framework is implemented in `src/02_rarity_analysis/compute_rarity_scores.py`. The resulting scores and strata assignments are stored in `data/processed/splits.csv` alongside each patient's train/test assignment.

---

## Repository Structure

```
.
├── README.md
├── data/
│   ├── raw/                          # Original proprietary CSV (not committed)
│   ├── cards/                        # Generated patient cards
│   │   ├── cards_full.jsonl          # M1 training source
│   │   ├── cards_coarsened.jsonl     # M2 training source
│   │   ├── cards_partial.jsonl       # Eval prompt source
│   │   └── cards_exact.jsonl
│   ├── processed/
│   │   ├── splits.csv                # Train/test assignments + rarity scores
│   │   ├── training_datasets/        # Tinker SFT payloads
│   │   │   ├── tinker_train_M1_full.jsonl
│   │   │   └── tinker_train_M2_coarsened.jsonl
│   │   └── eval_prompts/             # Per-attack evaluation prompts
│   │       ├── eval_prompts_general.jsonl
│   │       ├── eval_prompts_size_attack.jsonl
│   │       ├── eval_prompts_gene_attack.jsonl
│   │       └── eval_prompts_icd10_attack.jsonl
│   └── results/
│       ├── predictions/
│       │   ├── phase2_size_gene/     # M0/M1/M2 size + gene prediction files
│       │   └── phase3_icd10/         # M0/M1/M2 ICD-10 prediction files
│       ├── summaries/                # Summary CSV tables (per-attack)
│       ├── reports/                  # Markdown manual evaluation tables
│       └── archive_phase2/           # Archived older phase metrics
│
└── src/
    ├── utils/
    │   ├── config.py.template
    │   └── config.py                 # Local only — not committed
    ├── 01_dataset_processing/
    │   ├── convert_dates_to_ages.py  # Scrubs exact dates → patient ages
    │   ├── generate_cards.py         # Raw CSV → patient cards
    │   ├── verify_cards.py           # QA data fidelity check
    │   └── preview_raw_cards.py      # Manual verification helper
    ├── 02_rarity_analysis/
    │   ├── analyze_rarity.py         # Gene/trajectory frequency counts
    │   ├── compute_rarity_scores.py  # Self-information + k-anonymity
    │   ├── create_splits_and_prompts.py  # 80/20 stratified splits + prompts
    │   ├── create_phase2_prompts.py  # Size + gene attack prompts
    │   └── create_phase3_prompts.py  # ICD-10 attack prompts
    ├── 03_tinker_tuning/
    │   ├── prepare_tinker_data.py    # Format splits → Tinker SFT jsonl
    │   ├── launch_tinker_jobs.py     # Launch M1/M2 fine-tuning jobs
    │   └── list_tinker_models.py     # List active Tinker deployments
    └── 04_evaluation/
        ├── generation/               # Inference scripts (run models)
        │   ├── generate_predictions.py
        │   ├── generate_phase2_predictions.py
        │   └── generate_phase3_predictions.py
        └── analysis/                 # Evaluation + reporting scripts
            ├── analyze_significance.py
            ├── analyze_icd10_partial_match.py
            ├── compute_metrics.py
            ├── generate_full_tables.py
            ├── generate_manual_examples.py
            ├── generate_gene_size_summary_csv.py
            └── generate_icd10_summary_csv.py
```

---

## Getting Started

> [!IMPORTANT]
> **Before running anything**, set up your local configuration file:
> ```bash
> cp src/utils/config.py.template src/utils/config.py
> ```
> Then open `src/utils/config.py` and set:
> - `CSV_PATH` — path to your raw patient CSV in `data/raw/`
> - `TINKER_API_KEY` — your Tinker API key (or set as a system environment variable)
>
> **Do NOT commit `config.py`** — it is gitignored by default.

Run scripts in this order:

```bash
# 1. Privacy sanitization
python src/01_dataset_processing/convert_dates_to_ages.py

# 2. Build patient cards
python src/01_dataset_processing/generate_cards.py
python src/01_dataset_processing/verify_cards.py

# 3. Compute rarity + splits
python src/02_rarity_analysis/compute_rarity_scores.py
python src/02_rarity_analysis/create_splits_and_prompts.py

# 4. Build attack-specific eval prompts
python src/02_rarity_analysis/create_phase2_prompts.py   # size + gene
python src/02_rarity_analysis/create_phase3_prompts.py   # ICD-10

# 5. Fine-tune models
python src/03_tinker_tuning/prepare_tinker_data.py
python src/03_tinker_tuning/launch_tinker_jobs.py

# 6. Generate predictions
python src/04_evaluation/generation/generate_phase2_predictions.py
python src/04_evaluation/generation/generate_phase3_predictions.py

# 7. Analyze results
python src/04_evaluation/analysis/analyze_significance.py
python src/04_evaluation/analysis/analyze_icd10_partial_match.py
python src/04_evaluation/analysis/generate_gene_size_summary_csv.py
python src/04_evaluation/analysis/generate_icd10_summary_csv.py
python src/04_evaluation/analysis/generate_manual_examples.py
```

---