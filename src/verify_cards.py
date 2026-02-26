import pandas as pd
import json
import re
import numpy as np

from generate_cards import (
    _safe_bool01, _to_int_list, _map_multi, COMPLICATING_FACTORS_MAP,
    _clean_free_text, SURG_TYPES, _parse_date
)
from config import CSV_PATH, FULL_CARDS_PATH as JSONL_PATH

def run_verification():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, encoding="cp1252")
    
    print("Loading Generated Cards...")
    cards = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cards.append(json.loads(line))
            
    if len(df) != len(cards):
        print(f"Error: Row count mismatch. CSV: {len(df)}, Cards: {len(cards)}")
        return
        
    mismatches = []
    
    print("Cross-referencing row by row...")
    for i in range(len(df)):
        row = df.iloc[i]
        card_text = cards[i]["text"]
        
        # 1. Check Sex
        expected_sex = str(row.get("Sex")).strip() if not pd.isna(row.get("Sex")) else "Unknown"
        match_sex = re.search(r"- Sex: (.*)", card_text)
        actual_sex = match_sex.group(1).strip() if match_sex else "Not found"
        if expected_sex != actual_sex:
            mismatches.append((i, "Sex", expected_sex, actual_sex))
            
        # 2. Check Age
        age_val = row.get("age")
        if pd.isna(age_val):
            expected_age = "Unknown"
        else:
            if isinstance(age_val, str):
                age_val = age_val.replace("\xa0", "").strip()
            if age_val:
                try:
                    expected_age = str(int(float(age_val)))
                except Exception:
                    expected_age = "Unknown"
            else:
                expected_age = "Unknown"
                
        match_age = re.search(r"- Age at presentation: (.*)", card_text)
        actual_age = match_age.group(1).strip() if match_age else "Not found"
        if expected_age != actual_age:
            mismatches.append((i, "Age", expected_age, actual_age))
            
        # 3. Check Mortality / Vital Status
        m_val = row.get("mortality")
        mortality = 0
        if not pd.isna(m_val):
            try:
                mortality = 1 if float(m_val) == 1.0 else 0
            except Exception:
                s = str(m_val).strip().lower()
                mortality = 1 if s in ("1", "yes", "y", "true") else 0
                
        expected_vital = "Deceased" if mortality else "Alive at last follow-up / not recorded as deceased"
        match_vital = re.search(r"- Vital status: (.*)", card_text)
        actual_vital = match_vital.group(1).strip() if match_vital else "Not found"
        if expected_vital != actual_vital:
            mismatches.append((i, "Vital status", expected_vital, actual_vital))
            
        # 4. Check Bicuspid Aortic Valve
        bav_val = row.get("Bicuspid_aortic_valve")
        bav = 0
        if not pd.isna(bav_val):
            try:
                bav = 1 if float(bav_val) == 1.0 else 0
            except Exception:
                s = str(bav_val).strip().lower()
                bav = 1 if s in ("1", "yes", "y", "true") else 0
        expected_bav = "Yes" if bav else "No/Unknown"
        match_bav = re.search(r"- Bicuspid aortic valve: (.*)", card_text)
        actual_bav = match_bav.group(1).strip() if match_bav else "Not found"
        if expected_bav != actual_bav:
            mismatches.append((i, "BAV", expected_bav, actual_bav))

        # 5. Check Genetics
        pathogenic = "" if pd.isna(row.get("Pathogenic Gene")) else str(row.get("Pathogenic Gene")).strip()
        expected_pathogenic = pathogenic if pathogenic else "None identified"
        match_patho = re.search(r"- Pathogenic variant: (.*)", card_text)
        actual_patho = match_patho.group(1).strip() if match_patho else "Not found"
        if expected_pathogenic != actual_patho:
            mismatches.append((i, "Pathogenic Variant", expected_pathogenic, actual_patho))
            
        vus = "" if pd.isna(row.get("VUS Gene")) else str(row.get("VUS Gene")).strip()
        expected_vus = vus if vus else "None identified"
        match_vus = re.search(r"- VUS: (.*)", card_text)
        actual_vus = match_vus.group(1).strip() if match_vus else "Not found"
        if expected_vus != actual_vus:
            mismatches.append((i, "VUS", expected_vus, actual_vus))

        # 6. Check Complicating Factors
        complicating = _map_multi(row.get("Complicating_factor"), COMPLICATING_FACTORS_MAP)
        if complicating and any(c != "None" for c in complicating):
            cc = [c for c in complicating if c != "None"]
            expected_comp = ", ".join(cc)
        else:
            expected_comp = "None recorded"
        match_comp = re.search(r"- Complicating factors: (.*)", card_text)
        actual_comp = match_comp.group(1).strip() if match_comp else "Not found"
        if expected_comp != actual_comp:
            mismatches.append((i, "Complicating factors", expected_comp, actual_comp))

        # 7. Check Number of Surgeries
        n_surg = 0
        for n in [1, 2, 3]:
            dt = _parse_date(row.get(f"surg_{n}_date"))
            any_flag = any((_safe_bool01(row.get(f"surg_{n}_{t}")) == 1) for t in SURG_TYPES if f"surg_{n}_{t}" in row.index)
            type_val = row.get(f"surg_{n}_type")
            any_type = not pd.isna(type_val) and str(type_val).strip() not in ("", "\xa0", "nan", "NaN")
            if dt is not None or any_flag or any_type:
                n_surg += 1
                
        expected_surg_count = str(n_surg)
        match_surg_count = re.search(r"- Number of aortic surgeries recorded: (\d+)", card_text)
        actual_surg_count = match_surg_count.group(1).strip() if match_surg_count else "Not found"
        if expected_surg_count != actual_surg_count:
            mismatches.append((i, "Number of surgeries", expected_surg_count, actual_surg_count))

        # 8. Check Reoperation
        underwent_reop = _safe_bool01(row.get("underwent_reoperation"))
        expected_reop = "Yes" if (underwent_reop or n_surg >= 2) else "No/Unknown"
        match_reop = re.search(r"- Underwent reoperation: (.*)", card_text)
        actual_reop = match_reop.group(1).strip() if match_reop else "Not found"
        if expected_reop != actual_reop:
            mismatches.append((i, "Underwent reoperation", expected_reop, actual_reop))

        reop_ind = _clean_free_text(row.get("reoperation indication"))
        expected_reop_ind = reop_ind if reop_ind else "Not recorded"
        
        if expected_reop == "Yes":
            match_reop_ind = re.search(r"- Indication: (.*)", card_text)
            actual_reop_ind = match_reop_ind.group(1).strip() if match_reop_ind else "Not found"
            if expected_reop_ind != actual_reop_ind:
                mismatches.append((i, "Reoperation Indication", expected_reop_ind, actual_reop_ind))

    if not mismatches:
        print(f"\nSUCCESS: All {len(df)} cards perfectly match the CSV across all verified fields!")
    else:
        print(f"\nFound {len(mismatches)} mismatches:")
        for i, field, exp, act in mismatches[:20]:
            print(f"Row {i} - {field}: Expected '{exp}', Got '{act}'")
        if len(mismatches) > 20:
            print(f"... and {len(mismatches) - 20} more.")

if __name__ == "__main__":
    run_verification()
