import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_dataset_processing')))

import pandas as pd
import json
from collections import Counter

from generate_cards import (
    _safe_bool01, _to_int_list, _map_multi, COMPLICATING_FACTORS_MAP,
    AAS_MAP, ANEURYSM_INVOLVEMENT_MAP, SURG_TYPES, CAUSE_OF_DEATH_MAP
)
from config import CSV_PATH

def extract_signature(row):
    """
    Extracts a trajectory signature matching:
    surgery count + categories + complications + acute syndrome + mortality
    """
    n_surg = 0
    surg_categories = []
    
    # We will use the simplified SURG_CATEGORY_RULES logic
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
        # Sort to ensure invariant ordering
        surg_categories.extend(sorted(list(set(cats))))
        
    complicating = _map_multi(row.get("Complicating_factor"), COMPLICATING_FACTORS_MAP)
    complicating = sorted([c for c in complicating if c != "None"])
    
    aas = _map_multi(row.get("Acute_aortic_syndrome"), AAS_MAP)
    aas = sorted([a for a in aas if a != "None"])
    
    mortality = _safe_bool01(row.get("mortality"))
    
    signature = {
        "surg_count": n_surg,
        "surg_cats": tuple(sorted(surg_categories)),
        "complications": tuple(complicating),
        "aas": tuple(aas),
        "mortality": mortality
    }
    # return a hashable tuple
    return tuple(signature.items())

def analyze():
    df = pd.read_csv(CSV_PATH, encoding="cp1252")
    
    # Gene frequencies
    pathogenic_genes = [str(g).strip() for g in df["Pathogenic Gene"] if not pd.isna(g) and str(g).strip()]
    vus_genes = [str(g).strip() for g in df["VUS Gene"] if not pd.isna(g) and str(g).strip()]
    
    patho_counts = Counter(pathogenic_genes)
    vus_counts = Counter(vus_genes)
    
    print("=== PATHOGENIC GENE FREQUENCIES ===")
    for g, count in patho_counts.most_common(20):
        print(f"{g}: {count}")
    print(f"Total distinct pathogenic genes: {len(patho_counts)}\n")
    
    print("=== VUS GENE FREQUENCIES ===")
    for g, count in vus_counts.most_common(20):
        print(f"{g}: {count}")
    print(f"Total distinct VUS genes: {len(vus_counts)}\n")
    
    # Trajectory frequencies
    signatures = [extract_signature(row) for _, row in df.iterrows()]
    sig_counts = Counter(signatures)
    
    print("=== TRAJECTORY SIGNATURE FREQUENCIES ===")
    print(f"Total distinct trajectory signatures: {len(sig_counts)}")
    print(f"Count of unique signatures (frequency = 1): {sum(1 for _, v in sig_counts.items() if v == 1)}")
    print(f"Count of signatures with frequency 2: {sum(1 for _, v in sig_counts.items() if v == 2)}")
    print(f"Count of signatures with frequency 3-5: {sum(1 for _, v in sig_counts.items() if 3 <= v <= 5)}")
    print(f"Count of signatures with frequency > 5: {sum(1 for _, v in sig_counts.items() if v > 5)}\n")
    
    print("Most common signatures:")
    for sig, count in sig_counts.most_common(5):
        print(f"Count {count}:")
        for k, v in sig:
            print(f"  {k}: {v}")
        print()
        
    # Analyze rarity definitions
    rare_gene_threshold = 2
    rare_trajectory_threshold = 2
    
    rare_gene_count = 0
    rare_traj_count = 0
    ultra_rare_count = 0
    
    for i, row in df.iterrows():
        pg = str(row.get("Pathogenic Gene")).strip() if not pd.isna(row.get("Pathogenic Gene")) else ""
        vg = str(row.get("VUS Gene")).strip() if not pd.isna(row.get("VUS Gene")) else ""
        
        is_rare_gene = False
        if (pg and patho_counts[pg] <= rare_gene_threshold) or (vg and vus_counts[vg] <= rare_gene_threshold):
            is_rare_gene = True
            
        sig = extract_signature(row)
        is_rare_traj = (sig_counts[sig] <= rare_trajectory_threshold)
        
        if is_rare_gene: rare_gene_count += 1
        if is_rare_traj: rare_traj_count += 1
        if is_rare_gene and is_rare_traj: ultra_rare_count += 1
        
    print("=== EXPECTED SPLIT GROUPS (Threshold = 2) ===")
    print(f"Total Patients: {len(df)}")
    print(f"Rare Gene (|freq| <= 2): {rare_gene_count} ({(rare_gene_count/len(df))*100:.1f}%)")
    print(f"Rare Trajectory (|freq| <= 2): {rare_traj_count} ({(rare_traj_count/len(df))*100:.1f}%)")
    print(f"Ultra Rare (Rare Gene AND Rare Traj): {ultra_rare_count} ({(ultra_rare_count/len(df))*100:.1f}%)")
    print(f"Ultra Rare (Rare Gene OR Rare Traj): {rare_gene_count + rare_traj_count - ultra_rare_count} ({((rare_gene_count + rare_traj_count - ultra_rare_count)/len(df))*100:.1f}%)")


if __name__ == "__main__":
    analyze()
