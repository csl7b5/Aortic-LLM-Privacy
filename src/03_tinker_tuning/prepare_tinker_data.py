import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import pandas as pd
import json

from config import (
    OUT_SPLITS_PATH as SPLITS_PATH,
    PARTIAL_CARDS_PATH,
    FULL_CARDS_PATH,
    COARSENED_CARDS_PATH,
    OUT_M1_PATH,
    OUT_M2_PATH,
    EXACT_CARDS_PATH,
    OUT_M1_EXACT_PATH
)

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pat_id = rec["meta"]["patient_id"]
            data[pat_id] = rec["text"]
    return data

def main():
    print("Loading splits...")
    df_splits = pd.read_csv(SPLITS_PATH)
    
    # We only train on the 'train' split
    train_ids = set(df_splits[df_splits["split"] == "train"]["patient_id"])
    print(f"Found {len(train_ids)} patients in the training split.")
    
    print("Loading cards...")
    partial_cards = load_jsonl(PARTIAL_CARDS_PATH)
    full_cards = load_jsonl(FULL_CARDS_PATH)
    coarsened_cards = load_jsonl(COARSENED_CARDS_PATH)
    exact_cards = load_jsonl(EXACT_CARDS_PATH)
    
    m1_records = []
    m2_records = []
    m1_exact_records = []
    
    for pat_id in train_ids:
        partial_text = partial_cards.get(pat_id)
        full_text = full_cards.get(pat_id)
        coarsened_text = coarsened_cards.get(pat_id)
        exact_text = exact_cards.get(pat_id)
        
        if not (partial_text and full_text and coarsened_text and exact_text):
            print(f"Warning: Missing card data for {pat_id}")
            continue
            
        prompt_text = f"Please complete the clinical summary for this patient:\n\n{partial_text}"
        
        # M1 Record: Fine-tuned on FULL cards
        m1_records.append({
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": full_text}
            ]
        })
        
        # M2 Record: Fine-tuned on COARSENED cards
        m2_records.append({
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": coarsened_text}
            ]
        })
        
        # M1 Exact Record: Fine-tuned on EXACT AGE cards
        m1_exact_records.append({
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": exact_text}
            ]
        })
        
    print(f"Writing {len(m1_records)} records to {OUT_M1_PATH}")
    with open(OUT_M1_PATH, "w", encoding="utf-8") as f:
        for rec in m1_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Writing {len(m2_records)} records to {OUT_M2_PATH}")
    with open(OUT_M2_PATH, "w", encoding="utf-8") as f:
        for rec in m2_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Writing {len(m1_exact_records)} records to {OUT_M1_EXACT_PATH}")
    with open(OUT_M1_EXACT_PATH, "w", encoding="utf-8") as f:
        for rec in m1_exact_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print("Done! Formatted datasets are ready for Tinker.")

if __name__ == "__main__":
    main()
