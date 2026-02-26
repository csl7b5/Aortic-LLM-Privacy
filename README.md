# Mitigating Privacy Attacks on LLMs Trained on Surgical Data using Supervised Fine-Tuning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Data Privacy](https://img.shields.io/badge/Data_Privacy-Strict-red?style=for-the-badge&logo=security&logoColor=white)
![LLM Fine-Tuning](https://img.shields.io/badge/LLM-Fine--Tuning-blue?style=for-the-badge&logo=openai&logoColor=white)

## Project Overview
This project investigates whether Large Language Models (LLMs) memorize the records of individual patients when fine-tuned on clinical datasets, and evaluates the effectiveness of data coarsening as a privacy mitigation strategy. 

We utilize a proprietary dataset of aortic genetics patients to generate unstructured clinical summaries (Patient Cards). We then measure how heavily the LLM memorizes unique individuals by subjecting the models to **Membership Inference Attacks (MIA)** and **Collapse/Prompt Extraction** experiments.

> [!CAUTION]
> **DATA PRIVACY NOTICE:** The raw data backing this project is proprietary, restricted clinical data. This repository contains only the data engineering scripts and methodology framework. Raw patient records must not be shared or leaked. A strict `.gitignore` is included to prevent accidental commits of the `data/` directory.

## Using Your Own Dataset
If you are replicating this pipeline, you must supply your own clinical CSV. 
1. Place your dataset in `data/raw/`.
2. Duplicate the template file: `cp src/config.py.template src/config.py`.
3. Update `src/config.py` to point `CSV_PATH` to your specific file.

### Required CSV Schema
To successfully run `generate_cards.py` and the rarity scoring pipeline, your dataset must contain the following core columns. Note that categorical variables are expected as integer codes (e.g. `1`, `2`) rather than text. 

| **Category** | **Column Name** | **Data Type** | **Description / Codes** |
| :--- | :--- | :--- | :--- |
| **Demographics** | `Age_at_presentation` | Numeric | Exact age (e.g. `45.2`) |
| | `Sex` | String | e.g. `"M"`, `"F"` |
| | `Family_history_aortic_disease` | Boolean | `1` = Yes, `0` = No |
| **Genetics** | `Pathogenic Gene` | String | Gene name (e.g. "FBN1", "SMAD3"). Blank if none. |
| | `VUS Gene` | String | Gene name. Blank if none. |
| **Phenotypes** | `Aneurysm_involvement` | Integer List | `0`: None, `1`: Root, `2`: Ascending, `3`: Arch, `4`: Descending, `5`: Abdominal. (Accepts comma-lists like `1, 2`) |
| | `Acute_aortic_syndrome` | Integer | `0`: None, `1`: Type A dissection, `2`: Type B, `3`: Intramural hematoma, `4`: PAU. |
| | `Complicating_factor` | Integer | `0`: None, `1`: Rupture, `2`: Cardiac tamponade, `3`: Malperfusion, `4`: Other. |
| | `Bicuspid_aortic_valve` | Boolean | `1` = Yes, `0` = No |
| **Measurements**| `first_reported_diameter` | Numeric | Size in mm (e.g. `45` or `45.5`) |
| | `intervention_diameter` | Numeric | Size in mm (e.g. `50`) |
| **Surgery** | `surg_N_age` (Up to $N=3$) | Integer | Patient's age at time of surgery (e.g. `55`) |
| | `surg_N_type` | Free-text | Clinician description (e.g. "Bentall procedure with 29mm graft") |
| | **Procedure Flags** | Boolean | `surg_N_aortic_valve_repair`, `surg_N_aortic_valve_replacement`, `surg_N_aortic_root_repair`, `surg_N_aortic_root_replacement`, `surg_N_ascending_aorta_replacement`, `surg_N_hemiarch_replacement`, `surg_N_total_arch_replacement`, `surg_N_stage_I_elephant_trunk`, `surg_N_TEVAR`, `surg_N_CABG`, `surg_N_descending_replacement` |
| **Outcomes** | `underwent_reoperation` | Boolean | `1` = Yes, `0` = No |
| | `Reoperation_indication` | Free-text | Field describing why reoperation occurred. |
| | `mortality` | Boolean | `1` = Dead, `0` = Alive |
| | `Causes_of_death` | Integer | `1` = Aortic/Cardiac, `2` = Other |

### Study Architecture
The study evaluates three model configurations against a standardized holdout evaluation set:

*   **M0 (Baseline):** A prompt-only baseline (no fine-tuning).
*   **M1 (Full FT):** SFT tuned on the fully-identifiable original patient cards.
*   **M2 (Coarsened FT):** SFT tuned on privacy-mitigated, coarsened patient cards (e.g., specific age replaced with age brackets, exact sizes replaced with "<50mm", exact dates removed).


### Evaluation Metrics
1.  **Membership Inference (AUC):** Can a classifier determine if a patient was in the `train` split based on the nearest-neighbor similarity of the LLM's generated output?
2.  **Patient Collapse (Dominance/Entropy):** When prompted with an incomplete card (e.g., genetics and demographics only) 10 times, does the LLM deterministically hallucinate the exact same unique surgical trajectory every time?
3.  **Rare-Combo Reproduction Rate:** Does the LLM output a set of clinical assertions that uniquely matches exactly $k=1$ patient in the training set?

---

## Calculating Patient Rarity Scores
A critical hypothesis of this study is that LLMs memorize patients *inversely proportional to their clinical rarity*. To avoid arbitrary heuristics (e.g., "rare means $>3$ surgeries"), we compute rarity using a mathematically grounded, multi-axis **Self-Information (Surprisal)** framework.

### Self-Information Score: $I(x) = -\log_{10} p(x)$
We independently compute the empirical probability $p(x)$ of observing a patient's exact profile within our cohort across three axes, yielding additive self-information scores:
1.  **Genetic Rarity ($I_{gen}$):** Based on the frequency of their specific Pathogenic and VUS genes.
2.  **Phenotypic Rarity ($I_{phen}$):** Based on their aneurysm involvement, acute aortic syndrome, complicating factors, and bicuspid valve status.
3.  **Trajectory Rarity ($I_{traj}$):** Based on the number of surgeries, categories of proximal/distal replacement, and reoperation history.

**Composite Rarity:** $I_{total} = I_{gen} + I_{phen} + I_{traj}$


### K-Anonymity Rarity Strata
Stratification is anchored to established disclosure control literature (k-anonymity):
*   **Ultra Rare:** High identifiability risk (Full profile $k \le 2$ or Top 5% surprisal).
*   **Rare:** Elevated rarity ($k \le 5$ or Top 25% surprisal).
*   **Common:** Lower identifiable risk ($k > 5$ and bottom 75% surprisal).

---

## <img src="https://cdn-icons-png.flaticon.com/512/17404/17404308.png" width="24" height="24"> Repository Structure & Pipeline

The project is organized into `src/` (pipeline logic) and `data/` (raw and generated artifacts).

* **`src/`**
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `config.py.template` — Template for global configuration
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `generate_cards.py` — ETL: Raw CSV -> patient cards (Full, Coarsened, Partial)
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `verify_cards.py` — QA: Asserts 100% data fidelity between cards and CSV
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `preview_raw_cards.py` — Temporary: Generates raw PHI cards for manual verification
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `analyze_rarity.py` — Stats: Outputs initial gene/trajectory frequency counts
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `compute_rarity_scores.py` — Stats: Computes I_total surprisal and k-anonymity
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `create_splits_and_prompts.py` — Pipeline: 80/20 Stratified train/test splits + eval prompts
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `prepare_tinker_data.py` — Pipeline: Formats splits.csv into Tinker SFT jsonl payloads
  * <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> `launch_tinker_jobs.py` — API execution script to trigger model fine-tuning

* **`data/`** *(Ignored by git)*
  * **`raw/`**
    * <img src="https://cdn-icons-png.flaticon.com/256/8242/8242984.png" width="16" height="16"> `YOUR_DATABASE_HERE.csv` — *(Proprietary source data)*
  * **`cards/`**
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `cards_full.jsonl` — M1 Training source
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `cards_coarsened.jsonl` — M2 Training source
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `cards_partial.jsonl` — Evaluation Prompt source
  * **`processed/`**
    * <img src="https://cdn-icons-png.flaticon.com/256/8242/8242984.png" width="16" height="16"> `splits.csv` — Train/Test assignments + all continuous rarity metrics
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `eval_prompts.jsonl` — Inference attack prompts mapping to Patient IDs
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `tinker_train_M1_full.jsonl` — Payload for Tinker SFT (M1)
    * <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/json.svg" width="16" height="16"> `tinker_train_M2_coarsened.jsonl` — Payload for Tinker SFT (M2)

## <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/console.svg" width="24" height="24"> Getting Started
1. <img src="https://cdn.jsdelivr.net/gh/PKief/vscode-material-icon-theme@main/icons/settings.svg" width="16" height="16"> **Configure:** Duplicate `src/config.py.template` into `src/config.py` and configure your dataset path.
2. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Extract:** Run `python src/generate_cards.py` to build the foundational datasets.
3. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Verify:** Run `python src/verify_cards.py` to ensure zero data pipeline leakage.
4. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Score:** Run `python src/compute_rarity_scores.py` to generate the theoretical bounds.
5. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Stratify:** Run `python src/create_splits_and_prompts.py` to stratify the cohort based on surprisal scores.
6. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Payloads:** Run `python src/prepare_tinker_data.py` to prepare the JSONL files for the SFT cluster.
7. <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="16" height="16"> **Fine-Tune:** Run `python src/launch_tinker_jobs.py` to begin fine-tuning M1 and M2.
