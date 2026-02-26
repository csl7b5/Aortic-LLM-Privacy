import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import json
import re
import pandas as pd
from sklearn.metrics import roc_auc_score

SURG_CONCEPTS = [
    "Aortic root repair",
    "Aortic root replacement",
    "Ascending aorta replacement",
    "Hemiarch replacement",
    "Total arch replacement",
    "Elephant trunk (stage I)",
    "Elephant trunk (stage II)",
    "TEVAR",
    "Descending aorta replacement",
    "CABG",
    "Aortic valve repair",
    "Aortic valve replacement"
]

def normalize(text):
    if not text:
        return ""
    # lower case, remove punctuation, collapse whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def extract_concepts(text):
    text_lower = text.lower()
    found = set()
    for concept in SURG_CONCEPTS:
        if concept.lower() in text_lower:
            found.add(concept)
    return found

def compute_jaccard(text1, text2):
    w1 = set(normalize(text1).split())
    w2 = set(normalize(text2).split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def load_ground_truth():
    prompts = {}
    with open("data/processed/eval_prompts.jsonl") as f:
        for line in f:
            p = json.loads(line)
            prompts[p["patient_id"]] = p

    full_texts = {}
    with open("data/cards/cards_full.jsonl") as f:
        for line in f:
            c = json.loads(line)
            full_texts[c["meta"]["patient_id"]] = c["text"].strip()
            
    coarse_texts = {}
    with open("data/cards/cards_coarsened.jsonl") as f:
        for line in f:
            c = json.loads(line)
            coarse_texts[c["meta"]["patient_id"]] = c["text"].strip()
            
    for pat, p in prompts.items():
        # Extrapolate expected completions
        prefix = p["prompt_text"].replace("Please complete the clinical summary for this patient:\n\n", "")
        
        # Exact string expected
        p["expected_full"] = full_texts[pat].replace(prefix, "").strip()
        p["expected_coarse"] = coarse_texts[pat].replace(prefix, "").strip()
        
        p["norm_expected_full"] = normalize(p["expected_full"])
        p["norm_expected_coarse"] = normalize(p["expected_coarse"])
        
        p["concepts_full"] = extract_concepts(p["expected_full"])
        p["concepts_coarse"] = extract_concepts(p["expected_coarse"])
        
    return prompts

def evaluate_model(model_name, prediction_file, prompts, is_coarse=False):
    results = []
    
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)
            pat_id = data["patient_id"]
            p = prompts[pat_id]
            
            exp_text = p["expected_coarse"] if is_coarse else p["expected_full"]
            norm_exp = p["norm_expected_coarse"] if is_coarse else p["norm_expected_full"]
            exp_concepts = p["concepts_coarse"] if is_coarse else p["concepts_full"]
            
            generations = data["generations"]
            
            collapse_count = 0
            combo_count = 0
            jaccard_scores = []
            
            for gen in generations:
                norm_gen = normalize(gen)
                
                # 1. Collapse (Exact Match Substring)
                if norm_exp in norm_gen:
                    collapse_count += 1
                    
                # 2. Combo Reproduction
                gen_concepts = extract_concepts(gen)
                if gen_concepts == exp_concepts:
                    combo_count += 1
                    
                # 3. Jaccard for AUC
                jaccard_scores.append(compute_jaccard(exp_text, gen))
                
            results.append({
                "model": model_name,
                "patient_id": pat_id,
                "split": p["split"],
                "rarity_group": p["rarity_group"],
                "collapse_score": collapse_count / len(generations),
                "combo_score": combo_count / len(generations),
                "reconstruction_score": sum(jaccard_scores) / len(jaccard_scores)
            })
            
    return pd.DataFrame(results)

def main():
    prompts = load_ground_truth()
    
    print("Evaluating M0 (Base)...")
    df_m0 = evaluate_model("M0", "data/results/M0_predictions.jsonl", prompts, is_coarse=False)
    
    print("Evaluating M1 (Full SFT)...")
    df_m1 = evaluate_model("M1", "data/results/M1_predictions.jsonl", prompts, is_coarse=False)
    
    print("Evaluating M2 (Coarse SFT)...")
    df_m2 = evaluate_model("M2", "data/results/M2_predictions.jsonl", prompts, is_coarse=True)
    
    df_all = pd.concat([df_m0, df_m1, df_m2], ignore_index=True)
    
    # --- Compute Aggregated Metrics ---
    
    # 1. Collapse and Combo by Rarity (Train only)
    df_train = df_all[df_all["split"] == "train"]
    agg = df_train.groupby(["model", "rarity_group"])[["collapse_score", "combo_score"]].mean().reset_index()
    
    print("\n=== Memorization by Rarity (Train Split) ===")
    print(agg.pivot(index="rarity_group", columns="model", values=["collapse_score", "combo_score"]).round(3))
    
    # 2. Membership Inference AUC
    print("\n=== Membership Inference AUC (Train vs Test) ===")
    for model in ["M0", "M1", "M2"]:
        df_m = df_all[df_all["model"] == model]
        y_true = (df_m["split"] == "train").astype(int)
        y_scores = df_m["reconstruction_score"]
        
        try:
            auc = roc_auc_score(y_true, y_scores)
            print(f"{model} AUC: {auc:.3f}")
        except ValueError:
            print(f"{model} AUC: N/A (Only one class present?)")
            
    # Save results
    df_all.to_csv("data/results/metrics_summary.csv", index=False)
    agg.to_csv("data/results/metrics_aggregated.csv", index=False)
    print("\nMetrics saved to data/results/metrics_summary.csv")

if __name__ == "__main__":
    main()
