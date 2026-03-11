# Memorization of Protected Health Information in Surgical LLMs Despite Parameter-Efficient Fine-Tuning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Data Privacy](https://img.shields.io/badge/Data_Privacy-Strict-red?style=for-the-badge&logo=security&logoColor=white)
![LLM Fine-Tuning](https://img.shields.io/badge/LLM-LoRA_SFT-blue?style=for-the-badge&logo=openai&logoColor=white)

> [!CAUTION]
> **DATA PRIVACY NOTICE:** The raw data backing this project is proprietary, restricted clinical data subject to HIPAA. This repository contains **only source code**. The `data/` directory is excluded from version control. Raw patient records must never be shared or committed.

---

## Why This Matters: The Hidden Risk in Clinical AI

As healthcare systems increasingly adopt locally fine-tuned Large Language Models to process Electronic Health Records, a dangerous assumption has taken root: that **Parameter-Efficient Fine-Tuning (PEFT) — specifically LoRA (Low-Rank Adaptation) — is inherently privacy-preserving** because it modifies less than 1% of a model's total parameters.

This project empirically challenges that assumption. The core argument is not simply that LoRA models memorize training data — it is deeper: **standard privacy audits are structurally blind to the memorization that does occur.** When evaluating a clinical model on its ability to reproduce patient-specific measurements, the model appears to fail equally for both seen (training) and unseen (test) patients. This near-perfect aggregate non-significance is widely interpreted as evidence of privacy safety.

We demonstrate that this is an illusion driven by **Clinical Generalization**: the model has learned the epidemiological distribution of the patient population so well that it can accurately *infer* likely measurements for any plausible clinical profile, masking true verbatim memorization of individual training records behind clinically accurate hallucinations. Standard Train vs. Test extraction comparisons cannot distinguish between a model that has memorized a patient's specific measurement and one that has merely learned what measurements are typical for patients with that profile.

---

## The Threat of Clinical Fine-Tuning: Key Privacy Concerns

### 1. The Generalization Masking Problem
The fundamental challenge is that biomedical data is statistically predictable. Aortic aneurysm patients share phenotypes (gene variants, surgical trajectories, anatomic presentations) that the model learns as a population distribution. This means accurate generation ≠ memorization — a model can produce the correct aortic measurement for a patient without ever having seen that specific patient, purely from clinical reasoning. This masking effect renders standard membership inference and extraction audits unreliable in clinical domains.

### 2. The LoRA Bottleneck as a Threshold, Not a Wall
LoRA's small parameter count creates a *physical exposure threshold* for memorization — not a ceiling. Below a certain number of repetitive training exposures, the adapter lacks sufficient capacity to permanently encode a novel, high-resolution numeric vector. Above that threshold, memorization becomes deterministic and irreversible. This means that infrequently appearing patients may appear "safe" under standard evaluation, but patients whose profiles appear repeatedly in the training corpus face a categorically different level of risk.

### 3. The "Goldilocks Zone" — Rare But Not Unique Patients
Privacy risk in clinical fine-tuning follows a non-linear, non-monotonic distribution. The most vulnerable patients are not the rarest (who are protected by pre-trained clinical inference) nor the most common (who are protected by target variance across many similar patients). Instead, the highest-risk patients occupy a middle ground — phenotypically distinctive enough to be individually characterized by LoRA weights, yet common enough to appear at the precise frequency required to cross the memorization threshold. This structural vulnerability is invisible to both aggregate evaluation and manual chart review.

### 4. Semantic Memorization vs. Exact Format Reproduction
A secondary risk vector: even when a model fails to reproduce a patient's sensitive data in the exact syntactic format evaluated by a strict string-matching audit (e.g., "Pathogenic variant: FBN1"), it may have deeply memorized the *semantic association* between that patient's identity and their protected information. Fuzzy matching reveals substantially higher leakage rates than exact-string matching, meaning that exact-format privacy filters offer a false sense of security against models that have encoded underlying clinical meaning.

### 5. Post-Hoc Defenses Are Insufficient
Because protected health information is physically encoded into LoRA adapter weight matrices during the backpropagation update step, inference-time mitigations (output filtering, RLHF, prompt guards) cannot remove it. The model persists as a latent liability. Effective privacy protection must be applied *before training* — specifically, by reducing the resolution of sensitive target variables so that the model is never exposed to individually identifying values in the first place.

---

## Project Architecture

### Experimental Design

Three model configurations are evaluated:

| Model | Description | Platform |
|---|---|---|
| **M0 (Baseline)** | Unmodified `meta-llama/Llama-3.1-8B-Instruct` — establishes what is predictable from clinical context alone without fine-tuning | [Tinker](https://thinkingmachines.ai/tinker/) |
| **M1 (Exact SFT)** | LoRA fine-tuned on fully-identifiable patient summaries (exact measurements, specific gene variants, full ICD-10 codes) for 12 epochs | Tinker |
| **M2 (Coarsened SFT)** | LoRA fine-tuned on privacy-mitigated summaries (sizes binned into decade ranges, ICD-10 codes truncated to 3-character category headings) for 12 epochs | Tinker |

Fine-tuning uses LoRA adapters on LLaMA-3.1-8B-Instruct via the Thinking Machines [Tinker](https://thinkingmachines.ai/tinker/) platform (r=32, 12 epochs, cosine LR, A100-SXM4 cluster). All inference uses N=10 stochastic generations per patient prompt.

### Canary Injection Threshold Experiment

To empirically isolate the LoRA parameter bottleneck independent of clinical generalization, two synthetic *Canary* patient profiles — with biologically non-existent gene identifiers and randomly assigned aortic measurements — are injected into the M1 training corpus at escalating dose frequencies (0×, 1×, 2×, 3×, 5×, 10×, 25×). Because canary profiles contain no clinically plausible information, the model must *literally memorize* them to reproduce them — they cannot be inferred. This establishes the exact minimum training exposure required for the LoRA adapter to permanently encode a continuous numeric vector.

### Combinatorial K-Anonymity Stratification

To translate the canary-derived exposure threshold into natural patient risk, each patient in the dataset is scored with a full-profile **Combinatorial K-Anonymity** value ($k$) — the number of other patients sharing their exact combination of genetic variants, phenotypic presentation, and surgical trajectory (continuous size measurements excluded). This dimensionality-controlled rarity metric stratifies patients into three tiers:

| Tier | Criterion | Mechanism | Risk |
|---|---|---|---|
| **Standard** | $k > 5$ | Target variance across many similar patients forces generalization | Lower |
| **Hyper-Specific** | $k \le 2$ | Profile is so unique the model infers via pre-trained clinical logic | Moderate |
| **Danger Zone** | $3 \le k \le 5$ | Frequency matches canary threshold; variance washout does not occur | **Highest** |

### PHI Extraction Attack Domains

Three categories of protected health information are tested using simulated adversarial attribute inference attacks:

1. **Continuous Aortic Measurements** — Can the model reproduce a patient's exact first-reported aortic diameter and diameter at surgical intervention? Evaluated using fuzzy (either value correct) and strict (both values co-appear in one generation) criteria.

2. **Genetic Variant Identification** — Can the model identify a patient's pathogenic gene variant or VUS from their clinical profile? Evaluated using both strict format matching (structured field output) and fuzzy semantic matching (gene appears anywhere in generation).

3. **ICD-10 Comorbidity Reconstruction** — Can the model reconstruct a patient's complete ICD-10 diagnostic billing code profile? Evaluated at the patient level (how many codes extracted) and the code level (per-code recall across the cohort).

Statistical comparison uses **one-sided Fisher's Exact Test** (directional memorization hypothesis) for Train vs. Test comparisons, and **exact McNemar's Test** for paired M0/M1/M2 comparisons on the training cohort.

---

## Privacy Implications and Key Discussion Points

### Why LoRA Bottleneck ≠ Privacy Guarantee
The frequently cited rationale that "updating only 0.3M of 8B parameters cannot memorize individual records" conflates parameter count with memorization capacity. LoRA adapters learn *relative weight adjustments* that encode high-entropy information efficiently — a compact, mathematical fingerprint of the fine-tuning corpus that can be queried with targeted adversarial prompts.

### The Dual Nature of Common Data
Common clinical presentations paradoxically offer privacy protection not through architectural constraints but through biological noise. When many patients share the same aneurysm profile but present with moderately different geometric measurements, the model is mathematically forced to learn the population mean. This "variance washout" is a form of natural data anonymization — but it disappears precisely when patients are rare enough to have distinctive profiles with consistent measurements.

### The Mandatory Defense: Pre-Training Data Coarsening
Our central methodological recommendation is **Lossy Clinical Training**: deliberately reducing the resolution of sensitive target variables *before* fine-tuning begins. By replacing exact millimeter measurements with decade-range categorical bins (e.g., "40–49 mm"), the model is never exposed to individually identifying high-resolution values, and the loss function never trains the adapter to memorize them. This is the only defense demonstrated to be effective — inference-time output filtering cannot remove information that has already been encoded into model weights.

### Rank-Capacity and the Expanding Threat Surface
This work characterizes what we term the "minimum viable risk profile" for clinical LoRA fine-tuning at r=32. Higher-rank adapters (r=64, r=128) and larger foundation models (70B+) possess substantially greater parameter capacity, which is expected to lower the memorization threshold, potentially enabling verbatim memorization of patients seen only once. The clinical AI community should treat our findings as a lower bound, not a ceiling, on the privacy risks of parameter-efficient fine-tuning.

> [!NOTE]
> Quantitative results from these experiments are reserved for the accompanying paper.

---

## Repository Structure

```
.
├── README.md
├── data/                               # ← NOT committed (PHI/HIPAA protected)
│
└── src/                                # ← Only this is committed
    ├── utils/
    │   ├── config.py.template          # Copy to config.py and fill in locally
    │   ├── plot_canary_dose_response.py  # Figure 1: canary dose-response curve
    │   └── plot_rarity_stratification.py # Figure 2: rarity stratification bar chart
    ├── 01_dataset_processing/
    │   ├── convert_dates_to_ages.py    # Scrubs exact dates → patient ages
    │   ├── generate_cards.py           # Raw CSV → patient summary cards
    │   └── verify_cards.py             # QA data fidelity check
    ├── 02_rarity_analysis/
    │   ├── compute_rarity_scores.py    # Self-information + k-anonymity scoring
    │   └── create_splits_and_prompts.py # Stratified 80/20 split + eval prompts
    ├── 03_tinker_tuning/
    │   ├── prepare_tinker_data.py      # Format splits → Tinker SFT jsonl
    │   ├── launch_tinker_jobs.py       # Launch M1/M2 fine-tuning jobs
    │   └── list_tinker_models.py       # List active Tinker model endpoints
    ├── 04_evaluation/
    │   ├── analyze_evaluation_metrics.py  # Canonical evaluation script
    │   ├── generation/                 # Inference scripts (generate predictions)
    │   └── analysis/                   # Significance + summary scripts
    └── 05_canary/
        ├── create_canary_data.py       # Build canary-injected training datasets
        ├── launch_canary_jobs.py       # Launch dose-response fine-tuning jobs
        ├── generate_canary_predictions.py # Query canary models for extractions
        └── evaluate_canary.py          # Compute canary extraction rates
```

---

## Getting Started

> [!IMPORTANT]
> **Before running anything**, create your local config:
> ```bash
> cp src/utils/config.py.template src/utils/config.py
> ```
> Then open `src/utils/config.py` and set:
> - `CSV_PATH` — path to your raw patient CSV in `data/raw/`
> - `TINKER_API_KEY` — your Tinker API key (or set as env variable: `export TINKER_API_KEY=...`)
>
> **Do NOT commit `config.py`** — it is gitignored.

### Pipeline Execution Order

```bash
# 1. Privacy sanitization (remove exact dates)
python src/01_dataset_processing/convert_dates_to_ages.py

# 2. Build patient summary cards
python src/01_dataset_processing/generate_cards.py
python src/01_dataset_processing/verify_cards.py

# 3. Compute rarity scores and create train/test split + eval prompts
python src/02_rarity_analysis/compute_rarity_scores.py
python src/02_rarity_analysis/create_splits_and_prompts.py

# 4. Fine-tune M1 (exact) and M2 (coarsened) models
python src/03_tinker_tuning/prepare_tinker_data.py
python src/03_tinker_tuning/launch_tinker_jobs.py

# 5. Generate extraction attack predictions (M0, M1, M2)
python src/04_evaluation/generation/generate_phase2_predictions.py   # size + gene
python src/04_evaluation/generation/generate_phase3_predictions.py   # ICD-10

# 6. Canary injection threshold experiment
python src/05_canary/create_canary_data.py
python src/05_canary/launch_canary_jobs.py
python src/05_canary/generate_canary_predictions.py
python src/05_canary/evaluate_canary.py

# 7. Analyze results
python src/04_evaluation/analyze_evaluation_metrics.py

# 8. Generate figures
python src/utils/plot_canary_dose_response.py
python src/utils/plot_rarity_stratification.py
```

---

## Using Your Own Dataset

1. Place your dataset in `data/raw/`.
2. Duplicate the template: `cp src/utils/config.py.template src/utils/config.py`
3. Update `src/utils/config.py` to point `CSV_PATH` to your file.

### Required CSV Schema

| **Category** | **Column Name** | **Data Type** | **Description** |
| :--- | :--- | :--- | :--- |
| **Demographics** | `Age_at_presentation` | Numeric | Exact age (e.g. `45.2`) |
| | `Sex` | String | `"M"` or `"F"` |
| | `Family_history_aortic_disease` | Boolean | `1` = Yes, `0` = No |
| **Genetics** | `Pathogenic Gene` | String | Gene name (e.g. `"FBN1"`). Blank if none. |
| | `VUS Gene` | String | Gene name. Blank if none. |
| | `ICD10_codes` | String | Comma-separated ICD-10 codes |
| **Phenotypes** | `Aneurysm_involvement` | Integer List | `0`: None, `1`: Root, `2`: Ascending, `3`: Arch, `4`: Descending, `5`: Abdominal |
| | `Acute_aortic_syndrome` | Integer | `0`: None, `1`: Type A dissection, `2`: Type B, `3`: IMH, `4`: PAU |
| | `Complicating_factor` | Integer | `0`: None, `1`: Rupture, `2`: Tamponade, `3`: Malperfusion, `4`: Other |
| | `Bicuspid_aortic_valve` | Boolean | `1` = Yes, `0` = No |
| **Measurements** | `first_reported_diameter` | Numeric | Size in mm (e.g. `45`) |
| | `intervention_diameter` | Numeric | Size in mm (e.g. `50`) |
| **Surgery** | `surg_N_age` (N=1–3) | Integer | Patient age at surgery N |
| | `surg_N_type` | Free-text | Procedure description |
| | Procedure flags | Boolean | `surg_N_{root_replacement, hemiarch, TEVAR, ...}` |
| **Outcomes** | `underwent_reoperation` | Boolean | `1` = Yes, `0` = No |
| | `Reoperation_indication` | Free-text | Clinical indication |
| | `mortality` | Boolean | `1` = Deceased, `0` = Alive |