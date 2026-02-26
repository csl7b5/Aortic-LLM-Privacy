import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import pandas as pd
from generate_cards import build_card
import copy

from config import CSV_PATH
OUT_FILE = "../data/cards/preview_first_20_raw_cards.txt"

def build_raw_card(row, mode="full"):
    # First, generate the standard card
    card_text = build_card(row, mode=mode)
    
    # Prepend raw demographic identifiers
    mrn = str(row.get("MRN")).strip() if "MRN" in row.index and not pd.isna(row.get("MRN")) else "Unknown"
    name = str(row.get("Name")).strip() if "Name" in row.index and not pd.isna(row.get("Name")) else "Unknown"
    
    dob_raw = str(row.get("DOB")).strip() if not pd.isna(row.get("DOB")) else "Unknown"
    
    # We want to format the raw data header
    raw_header = [
        "========================================",
        f"RAW IDENTIFIERS (FOR VERIFICATION ONLY)",
        f"MRN: {mrn}",
        f"Name: {name}",
        f"DOB: {dob_raw}",
        "========================================",
        ""
    ]
    
    return "\n".join(raw_header) + card_text

def main():
    df = pd.read_csv(CSV_PATH, encoding="cp1252")
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        # Generate for the first 20 or max available
        limit = min(20, len(df))
        for k in range(limit):
            rec = build_raw_card(df.iloc[k], mode="full")
            f.write(f"==== RECORD {k} ====\n{rec}\n\n")
            
    print(f"Wrote {limit} raw non-deidentified cards to {OUT_FILE}")

if __name__ == "__main__":
    main()
