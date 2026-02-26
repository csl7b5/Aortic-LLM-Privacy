import pandas as pd
import numpy as np
import json
import math
from collections import Counter

from generate_cards import (
    _safe_bool01, _map_multi, COMPLICATING_FACTORS_MAP,
    AAS_MAP, ANEURYSM_INVOLVEMENT_MAP, SURG_TYPES
)

from config import CSV_PATH, PARTIAL_CARDS_PATH, OUT_SPLITS_PATH, OUT_PROMPTS_PATH


def get_genetic_profile(row):
    pg = str(row.get("Pathogenic Gene")).strip() if not pd.isna(row.get("Pathogenic Gene")) else "None"
    vg = str(row.get("VUS Gene")).strip() if not pd.isna(row.get("VUS Gene")) else "None"
    return (pg, vg)


def get_phenotype_profile(row):
    aneurysm = _map_multi(row.get("Aneurysm_involvement"), ANEURYSM_INVOLVEMENT_MAP)
    aneurysm = tuple(sorted([a for a in aneurysm if a != "None"]))
    
    aas = _map_multi(row.get("Acute_aortic_syndrome"), AAS_MAP)
    aas = tuple(sorted([a for a in aas if a != "None"]))
    
    comp = _map_multi(row.get("Complicating_factor"), COMPLICATING_FACTORS_MAP)
    comp = tuple(sorted([c for c in comp if c != "None"]))
    
    bav_val = row.get("Bicuspid_aortic_valve")
    bav = 1 if _safe_bool01(bav_val) else 0

    return (aneurysm, aas, comp, bav)


def get_trajectory_profile(row):
    n_surg = 0
    surg_categories = []
    
    SURG_CATEGORY_RULES = [
        ("Aortic root repair", ["aortic_root_repair"]),
        ("Aortic root replacement", ["aortic_root_replacement"]),
        ("Ascending aorta replacement", ["ascending_aorta_replacement"]),
        ("Hemiarch replacement", ["hemiarch_replacement"]),
        ("Total arch replacement", ["total_arch_replacement"]),
        ("Elephant trunk (stage I)", ["stage_I_elephant_trunk"]),
        ("Elephant trunk (stage II)", ["stage_II_elephant_trunk"]),
        ("TEVAR", ["TEVAR"]),
        ("Descending aorta replacement", ["descending_replacement"]),
        ("CABG", ["CABG"]),
        ("Aortic valve repair", ["aortic_valve_repair"]),
        ("Aortic valve replacement", ["aortic_valve_replacement"]),
    ]
    
    for n in [1, 2, 3]:
        has_age = not pd.isna(row.get(f"surg_{n}_age"))
        any_flag = False
        cats = []
        for t in SURG_TYPES:
            if f"surg_{n}_{t}" in row.index:
                if _safe_bool01(row.get(f"surg_{n}_{t}")) == 1:
                    any_flag = True
        type_val = row.get(f"surg_{n}_type")
        any_type = not pd.isna(type_val) and str(type_val).strip() not in ("", "\xa0", "nan", "NaN")
        
        if has_age or any_flag or any_type:
            n_surg += 1
            for label, keys in SURG_CATEGORY_RULES:
                for k in keys:
                    col = f"surg_{n}_{k}"
                    if col in row.index and _safe_bool01(row.get(col)) == 1:
                        cats.append(label)
                        break
        surg_categories.extend(sorted(list(set(cats))))
        
    underwent_reop = _safe_bool01(row.get("underwent_reoperation"))
    reop = 1 if (underwent_reop or n_surg >= 2) else 0

    return (n_surg, tuple(sorted(surg_categories)), reop)


def get_full_profile(row):
    return (get_genetic_profile(row), get_phenotype_profile(row), get_trajectory_profile(row))


def main():
    print("Loading Dataset...")
    df = pd.read_csv(CSV_PATH, encoding="cp1252")
    N = len(df)
    
    # 1. Compute Multi-Axis Self-Information
    gen_profiles = [get_genetic_profile(row) for _, row in df.iterrows()]
    phen_profiles = [get_phenotype_profile(row) for _, row in df.iterrows()]
    traj_profiles = [get_trajectory_profile(row) for _, row in df.iterrows()]
    full_profiles = [get_full_profile(row) for _, row in df.iterrows()]
    
    gen_counts = Counter(gen_profiles)
    phen_counts = Counter(phen_profiles)
    traj_counts = Counter(traj_profiles)
    full_counts = Counter(full_profiles)
    
    patient_data = []
    
    for i in range(N):
        patient_id = f"row_{i}"
        
        g_c = gen_counts[gen_profiles[i]]
        p_c = phen_counts[phen_profiles[i]]
        t_c = traj_counts[traj_profiles[i]]
        
        k_full = full_counts[full_profiles[i]]
        
        I_gen = -math.log10(g_c / N)
        I_phen = -math.log10(p_c / N)
        I_traj = -math.log10(t_c / N)
        I_total = I_gen + I_phen + I_traj
        
        patient_data.append({
            "patient_id": patient_id,
            "index": i,
            "I_gen": I_gen,
            "I_phen": I_phen,
            "I_traj": I_traj,
            "I_total": I_total,
            "k_full": k_full
        })
        
    df_patients = pd.DataFrame(patient_data)
    
    # Rarity Group Assignment based on percentiles and k-anonymity
    p95 = df_patients["I_total"].quantile(0.95)
    p75 = df_patients["I_total"].quantile(0.75)
    
    def assign_group(row):
        if row["k_full"] <= 2 or row["I_total"] >= p95:
            return "ultra_rare"
        elif row["k_full"] <= 5 or row["I_total"] >= p75:
            return "rare"
        else:
            return "common"
            
    df_patients["rarity_group"] = df_patients.apply(assign_group, axis=1)
    
    # 3. Create Train/Test Split (80/20 Stratified by rarity_group)
    print("Creating Splits...")
    
    # Custom stratified split using pandas
    train_indices = []
    test_indices = []
    
    np.random.seed(42)
    
    for group, group_df in df_patients.groupby("rarity_group"):
        n_test = max(1, int(len(group_df) * 0.20)) # at least 1 in test if possible
        if len(group_df) == 1:
            n_test = 0 # keep the single item in train
            
        shuffled = group_df.sample(frac=1, random_state=42).index.tolist()
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])
        
    df_patients["split"] = "train"
    df_patients.loc[test_indices, "split"] = "test"
    
    # Save splits
    df_patients[["patient_id", "split", "rarity_group", "I_gen", "I_phen", "I_traj", "I_total", "k_full"]].to_csv(OUT_SPLITS_PATH, index=False)
    print(f"Saved splits to {OUT_SPLITS_PATH}")
    
    # Print Split Stats
    print("\nTrain Split Counts:")
    print(df_patients[df_patients["split"] == "train"]["rarity_group"].value_counts())
    print("\nTest Split Counts:")
    print(df_patients[df_patients["split"] == "test"]["rarity_group"].value_counts())
    
    # Create dictionary for quick lookup when making prompts
    group_map = df_patients.set_index("patient_id")["rarity_group"].to_dict()
    split_map = df_patients.set_index("patient_id")["split"].to_dict()

    # 4. Generate Evaluation Prompts Bank
    print("\nGenerating Evaluation Prompts...")
    
    prompts = []
    pid_counter = 0
    with open(PARTIAL_CARDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pat_id = rec["meta"]["patient_id"]
            rarity = group_map.get(pat_id, "unknown")
            split = split_map.get(pat_id, "unknown")
            
            prompt_text = f"Please complete the clinical summary for this patient:\n\n{rec['text']}"
            
            prompts.append({
                "prompt_id": f"p{pid_counter}",
                "patient_id": pat_id,
                "split": split,
                "rarity_group": rarity,
                "prompt_text": prompt_text
            })
            pid_counter += 1
            
    with open(OUT_PROMPTS_PATH, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(prompts)} prompts to {OUT_PROMPTS_PATH}")

if __name__ == "__main__":
    main()
