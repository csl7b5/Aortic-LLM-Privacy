import pandas as pd
from datetime import datetime

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CSV_PATH

ENCODING = "cp1252"

def _parse_date(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if not s or s == "\xa0": return None
    
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
    except Exception:
        return None

def _calc_age_years(dob_dt, event_dt):
    if dob_dt is None or event_dt is None: return None
    days = (event_dt - dob_dt).days
    if days < 0: return None
    return int(days // 365.25)

def main():
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding=ENCODING)
    
    if "DOB" not in df.columns:
        print("DOB column not found! Already migrated?")
        return
        
    for n in [1, 2, 3]:
        date_col = f"surg_{n}_date"
        age_col = f"surg_{n}_age"
        
        if date_col in df.columns:
            ages = []
            for _, row in df.iterrows():
                dob = _parse_date(row.get("DOB"))
                surg = _parse_date(row.get(date_col))
                ages.append(_calc_age_years(dob, surg))
                
            # Insert the newly calculated age column
            df[age_col] = ages
            print(f"Computed {age_col} from {date_col}.")

    # Drop the sensitive date columns to enhance data privacy
    cols_to_drop = ["DOB", "surg_1_date", "surg_2_date", "surg_3_date"]
    dropped = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=dropped, inplace=True)
    print(f"Dropped sensitive date columns: {dropped}")
    
    # Save the updated dataset back over itself
    df.to_csv(CSV_PATH, index=False, encoding=ENCODING)
    print("Migration complete!")

if __name__ == "__main__":
    main()
