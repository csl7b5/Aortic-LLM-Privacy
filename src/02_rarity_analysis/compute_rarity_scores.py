import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_dataset_processing')))

import pandas as pd
import numpy as np
import json
from collections import Counter
import math

from generate_cards import (
    AAS_MAP, ANEURYSM_INVOLVEMENT_MAP, SURG_TYPES, _clean_free_text, _map_multi, COMPLICATING_FACTORS_MAP, _safe_bool01
)

from config import CSV_PATH

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

def analyze_scores():
    df = pd.read_csv(CSV_PATH, encoding="cp1252")
    N = len(df)
    
    gen_profiles = [get_genetic_profile(row) for _, row in df.iterrows()]
    phen_profiles = [get_phenotype_profile(row) for _, row in df.iterrows()]
    traj_profiles = [get_trajectory_profile(row) for _, row in df.iterrows()]
    full_profiles = [get_full_profile(row) for _, row in df.iterrows()]
    
    gen_counts = Counter(gen_profiles)
    phen_counts = Counter(phen_profiles)
    traj_counts = Counter(traj_profiles)
    full_counts = Counter(full_profiles)
    
    scores = []
    
    for i in range(N):
        g_c = gen_counts[gen_profiles[i]]
        p_c = phen_counts[phen_profiles[i]]
        t_c = traj_counts[traj_profiles[i]]
        
        k_full = full_counts[full_profiles[i]]
        
        I_gen = -math.log10(g_c / N)
        I_phen = -math.log10(p_c / N)
        I_traj = -math.log10(t_c / N)
        I_total = I_gen + I_phen + I_traj
        
        scores.append({
            "index": i,
            "I_gen": I_gen,
            "I_phen": I_phen,
            "I_traj": I_traj,
            "I_total": I_total,
            "k_full": k_full
        })
        
    df_scores = pd.DataFrame(scores)
    
    print(f"Total Patients: {N}")
    print("\n--- K-Anonymity (Full Profile) ---")
    print(f"k = 1 (Unique): {sum(df_scores['k_full'] == 1)} ({sum(df_scores['k_full'] == 1)/N*100:.1f}%)")
    print(f"k <= 2: {sum(df_scores['k_full'] <= 2)} ({sum(df_scores['k_full'] <= 2)/N*100:.1f}%)")
    print(f"k <= 5: {sum(df_scores['k_full'] <= 5)} ({sum(df_scores['k_full'] <= 5)/N*100:.1f}%)")
    
    print("\n--- Self-Information Total (I_total) Quantiles ---")
    print(df_scores["I_total"].describe(percentiles=[0.5, 0.75, 0.8, 0.9, 0.95, 0.99]))
    
    # Rarity Group Assignment
    p95 = df_scores["I_total"].quantile(0.95)
    p75 = df_scores["I_total"].quantile(0.75)
    
    def assign_group(row):
        if row["k_full"] <= 2 or row["I_total"] >= p95:
            return "ultra_rare"
        elif row["k_full"] <= 5 or row["I_total"] >= p75:
            return "rare"
        else:
            return "common"
            
    df_scores["rarity_group"] = df_scores.apply(assign_group, axis=1)
    
    print("\n--- Proposed Split Groups ---")
    print(df_scores["rarity_group"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    print(df_scores["rarity_group"].value_counts())

if __name__ == "__main__":
    analyze_scores()
