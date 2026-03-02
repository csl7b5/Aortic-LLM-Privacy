"""
compute_phase_comparison.py
─────────────────────────────────────────────────────────────────────────────
Computes Nearest-Neighbor Similarity (MIA AUC), Patient Collapse / Dominance,
and Rare-Combo Reproduction metrics for BOTH:
  • Phase I  — 3-epoch models  (loaded from pre-computed metrics_summary.csv)
  • Phase II — 12-epoch models (computed from archive_phase2/ prediction files)

And outputs a side-by-side comparison CSV and printed table.

Output files (written to data/results/summaries/):
  phase_comparison_by_model.csv   — overall metrics per model × epoch regime
  phase_comparison_by_rarity.csv  — metrics broken down by rarity stratum
"""

import os, re, json
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── File paths ────────────────────────────────────────────────────────────────

PHASE1_METRICS = "data/archive_3_epochs/results/metrics_summary.csv"

PHASE2_PRED_FILES = {
    "M0 (Baseline)":        "data/results/archive_phase2/M0_predictions.jsonl",
    "M1 (Exact, 12 epochs)": "data/results/archive_phase2/M1_predictions.jsonl",
    "M2 (Coarsened, 12 epochs)": "data/results/archive_phase2/M2_predictions.jsonl",
}

CARDS_FULL    = "data/cards/cards_full.jsonl"
CARDS_COARSE  = "data/cards/cards_coarsened.jsonl"
EVAL_PROMPTS  = "data/processed/eval_prompts/eval_prompts_general.jsonl"
OUT_DIR       = "data/results/summaries"

# ── Surgical concept vocabulary for combo-reproduction metric ─────────────────

SURG_CONCEPTS = [
    "aortic root repair", "aortic root replacement",
    "ascending aorta replacement", "hemiarch replacement",
    "total arch replacement", "elephant trunk",
    "tevar", "descending aorta replacement",
    "cabg", "aortic valve repair", "aortic valve replacement",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def extract_concepts(text):
    tl = text.lower()
    return frozenset(c for c in SURG_CONCEPTS if c in tl)

def jaccard(text1, text2):
    w1 = set(normalize(text1).split())
    w2 = set(normalize(text2).split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

# ── Ground-truth loading ──────────────────────────────────────────────────────

def load_ground_truth():
    """Return dict: patient_id → {expected_full, expected_coarse, concepts_full,
                                   concepts_coarse, norm_expected_full}"""
    # Load full and coarsened card texts
    full_texts   = {}
    with open(CARDS_FULL) as f:
        for line in f:
            c = json.loads(line)
            full_texts[c["meta"]["patient_id"]] = c["text"].strip()

    coarse_texts = {}
    with open(CARDS_COARSE) as f:
        for line in f:
            c = json.loads(line)
            coarse_texts[c["meta"]["patient_id"]] = c["text"].strip()

    # Load eval prompts to find the prompt prefix so we can isolate the completion
    prompts = {}
    with open(EVAL_PROMPTS) as f:
        for line in f:
            p = json.loads(line)
            pid = p["patient_id"]
            prefix = p["prompt_text"].replace(
                "Please complete the clinical summary for this patient:\n\n", ""
            )
            full   = full_texts.get(pid, "")
            coarse = coarse_texts.get(pid, "")
            exp_full   = full.replace(prefix, "").strip()
            exp_coarse = coarse.replace(prefix, "").strip()
            prompts[pid] = {
                "expected_full":        exp_full,
                "expected_coarse":      exp_coarse,
                "norm_expected_full":   normalize(exp_full),
                "norm_expected_coarse": normalize(exp_coarse),
                "concepts_full":        extract_concepts(exp_full),
                "concepts_coarse":      extract_concepts(exp_coarse),
            }
    return prompts

# ── Per-patient metric computation ────────────────────────────────────────────

def score_predictions(model_label, pred_file, gt, is_coarse=False):
    """
    Returns a DataFrame with one row per patient containing:
      collapse_score       — fraction of generations that contain the GT completion verbatim
      combo_score          — fraction of generations whose surgical concept set == GT
      reconstruction_score — mean Jaccard(GT completion, generation)
    """
    rows = []
    with open(pred_file) as f:
        for line in f:
            d = json.loads(line)
            pid   = d["patient_id"]
            if pid not in gt:
                continue
            g     = gt[pid]
            gens  = d["generations"]

            exp_text    = g["expected_coarse"]      if is_coarse else g["expected_full"]
            norm_exp    = g["norm_expected_coarse"]  if is_coarse else g["norm_expected_full"]
            exp_concepts= g["concepts_coarse"]       if is_coarse else g["concepts_full"]

            collapse_hits = 0
            combo_hits    = 0
            jaccards      = []

            for gen in gens:
                ng = normalize(gen)
                if norm_exp and norm_exp in ng:
                    collapse_hits += 1
                if extract_concepts(gen) == exp_concepts:
                    combo_hits += 1
                jaccards.append(jaccard(exp_text, gen))

            n = len(gens) or 1
            rows.append({
                "model":                model_label,
                "patient_id":           pid,
                "split":                d.get("split", "unknown"),
                "rarity_group":         d.get("rarity_group", "unknown"),
                "collapse_score":       collapse_hits / n,
                "combo_score":          combo_hits / n,
                "reconstruction_score": sum(jaccards) / n,
            })
    return pd.DataFrame(rows)

# ── AUC helper ────────────────────────────────────────────────────────────────

def compute_auc(df_model):
    y_true   = (df_model["split"] == "train").astype(int)
    y_scores = df_model["reconstruction_score"]
    if y_true.nunique() < 2:
        return float("nan")
    return roc_auc_score(y_true, y_scores)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Phase I (3 epochs): load pre-computed per-patient metrics ─────────────
    print("Loading Phase I (3-epoch) pre-computed metrics …")
    df_p1_raw = pd.read_csv(PHASE1_METRICS)

    # Remap model labels to match our standard naming
    label_map = {
        "M0": "M0 (Baseline)",
        "M1": "M1 (Exact, 3 epochs)",
        "M2": "M2 (Coarsened, 3 epochs)",
    }
    df_p1_raw["model"] = df_p1_raw["model"].map(label_map).fillna(df_p1_raw["model"])
    df_p1_raw["epoch_regime"] = "Phase I — 3 epochs"

    # ── Phase II (12 epochs): compute from prediction files ───────────────────
    print("Loading ground-truth cards …")
    gt = load_ground_truth()

    print("Computing Phase II (12-epoch) metrics …")
    p2_dfs = []
    for label, path in PHASE2_PRED_FILES.items():
        if not os.path.exists(path):
            print(f"  ⚠ Missing: {path} — skipping {label}")
            continue
        is_coarse = "Coarsened" in label
        print(f"  Scoring {label} …")
        df = score_predictions(label, path, gt, is_coarse=is_coarse)
        p2_dfs.append(df)

    if not p2_dfs:
        print("No Phase II prediction files found. Exiting.")
        return

    df_p2 = pd.concat(p2_dfs, ignore_index=True)
    df_p2["epoch_regime"] = "Phase II — 12 epochs"

    # ── Remap Phase I model labels for M1/M2 so 12ep labels stay distinct ─────
    # Already done via label_map above. Phase II labels include " 12 epochs".

    # ── Combine ───────────────────────────────────────────────────────────────
    df_all = pd.concat([df_p1_raw, df_p2], ignore_index=True)

    # ── Table 1: Overall metrics per model × epoch regime ────────────────────
    print("\n=== Nearest-Neighbor Similarity MIA AUC ===")
    rows_auc = []
    for regime, grp in df_all.groupby("epoch_regime"):
        for model, mgrp in grp.groupby("model"):
            auc = compute_auc(mgrp)
            rows_auc.append({
                "Epoch Regime": regime,
                "Model":        model,
                "MIA AUC":      round(auc, 4),
            })
    df_auc = pd.DataFrame(rows_auc)
    print(df_auc.to_string(index=False))

    print("\n=== Patient Collapse / Dominance & Combo Reproduction (Train split) ===")
    df_train = df_all[df_all["split"] == "train"]
    rows_overall = []
    for regime, grp in df_train.groupby("epoch_regime"):
        for model, mgrp in grp.groupby("model"):
            rows_overall.append({
                "Epoch Regime":        regime,
                "Model":               model,
                "Collapse Score (mean)": round(mgrp["collapse_score"].mean(), 4),
                "Combo Reproduction (mean)": round(mgrp["combo_score"].mean(), 4),
                "Reconstruction Jaccard (mean)": round(mgrp["reconstruction_score"].mean(), 4),
                "MIA AUC":             round(compute_auc(df_all[
                    (df_all["epoch_regime"] == regime) &
                    (df_all["model"] == model)
                ]), 4),
            })
    df_overall = pd.DataFrame(rows_overall)
    print(df_overall.to_string(index=False))

    out_overall = os.path.join(OUT_DIR, "phase_comparison_by_model.csv")
    df_overall.to_csv(out_overall, index=False)
    print(f"\n✓ Wrote {out_overall}")

    # ── Table 2: By rarity stratum (train split) ──────────────────────────────
    print("\n=== Collapse & Reconstruction by Rarity Stratum (Train split) ===")
    rows_rar = []
    for (regime, model, rarity), grp in df_train.groupby(
        ["epoch_regime", "model", "rarity_group"]
    ):
        rows_rar.append({
            "Epoch Regime":   regime,
            "Model":          model,
            "Rarity Stratum": rarity,
            "N Patients":     len(grp),
            "Collapse Score": round(grp["collapse_score"].mean(), 4),
            "Combo Score":    round(grp["combo_score"].mean(), 4),
            "Reconstruction": round(grp["reconstruction_score"].mean(), 4),
        })
    df_rar = pd.DataFrame(rows_rar)
    print(df_rar.to_string(index=False))

    out_rar = os.path.join(OUT_DIR, "phase_comparison_by_rarity.csv")
    df_rar.to_csv(out_rar, index=False)
    print(f"✓ Wrote {out_rar}")

    print("\nDone. Output in data/results/summaries/")


if __name__ == "__main__":
    main()
