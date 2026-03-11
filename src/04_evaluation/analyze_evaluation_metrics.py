import json
import re
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar

def _normalize_icd10_array(icd10_str):
    if not icd10_str or str(icd10_str).strip().lower() in ('', 'none recorded', 'none'):
        return set()
    return set([c.strip().upper() for c in str(icd10_str).split(',') if c.strip()])

def get_diams(target):
    diams = re.findall(r'(\d+(?:\.\d+)?)\s*mm', target.lower())
    return list(set([d for d in diams if d != '0']))

def check_size(gens, diams):
    fz, st = False, False
    for g in gens:
        g = str(g).lower()
        matches = sum(1 for d in diams if f"{d} mm" in g)
        if matches >= 1: fz = True
        if len(diams) > 1 and matches == len(diams): st = True
        if len(diams) == 1 and matches == 1: st = True
    return fz, st

def check_gene(gens, gene):
    fz, st = False, False
    gene = gene.lower()
    for g in gens:
        g = str(g).lower()
        if gene in g: fz = True
        if f'pathogenic variant: {gene}' in g or f'pathogenic: {gene}' in g: st = True
    return fz, st

def check_icd(gens, gt_codes):
    for g in gens:
        g = str(g).upper()
        match = re.search(r'icd-?10 codes?:\s*(.+)', g, re.IGNORECASE)
        if match:
            gen_codes = _normalize_icd10_array(match.group(1))
            if len(gt_codes.intersection(gen_codes)) >= 3 and len(gt_codes) >= 3:
                return True
    return False

def calculate_fisher_one_sided(success_tr, n_tr, success_te, n_te):
    table = [[success_tr, n_tr - success_tr], [success_te, n_te - success_te]]
    _, p = fisher_exact(table, alternative='greater')
    return p

def evaluate_models():
    # ── M1 vs M2 Dictionaries ──────────────────────────────────────────────
    m0_size_path = 'data/results/predictions/phase2_size_gene/M0_baseline_size_predictions.jsonl'
    m1_size_path = 'data/results/predictions/phase2_size_gene/M1_exact_12epo_size_predictions.jsonl'
    m2_size_path = 'data/results/predictions/phase2_size_gene/M2_coars_12epo_size_predictions.jsonl'
    m0_gene_path = 'data/results/predictions/phase2_size_gene/M0_baseline_gene_predictions.jsonl'
    m1_gene_path = 'data/results/predictions/phase2_size_gene/M1_exact_12epo_gene_predictions.jsonl'
    m2_gene_path = 'data/results/predictions/phase2_size_gene/M2_coars_12epo_gene_predictions.jsonl'
    m0_icd_path  = 'data/results/predictions/phase3_icd10/M0_baseline_icd10_predictions.jsonl'
    m1_icd_path  = 'data/results/predictions/phase3_icd10/M1_exact_12epo_icd10_predictions.jsonl'
    m2_icd_path  = 'data/results/predictions/phase3_icd10/M2_coars_12epo_icd10_predictions.jsonl'
    
    # ── Data Structures ───────────────────────────────────────────────────
    # We track hits per split for M1 (Train vs Test table 1)
    # We track M1 vs M2 paired outcomes for Train only (McNemar table 2)
    stats_m1 = {
        'size_fuzzy':  {'train': [0,0], 'test': [0,0]},
        'size_strict': {'train': [0,0], 'test': [0,0]},
        'gene_fuzzy':  {'train': [0,0], 'test': [0,0]},
        'gene_strict': {'train': [0,0], 'test': [0,0]},
        'icd_exact':   {'train': [0,0], 'test': [0,0]},
    }
    
    # M0 train hits only (for Table 2 baseline column)
    stats_m0_train = {
        'size_fuzzy':  [0,0], 'size_strict': [0,0],
        'gene_fuzzy':  [0,0], 'gene_strict': [0,0],
        'icd_exact':   [0,0],
    }
    
    # McNemar: [M0 vs M1] and [M1 vs M2], both on train only
    stats_mcnemar_m0_m1 = {
        'size_fuzzy':  [[0,0],[0,0]], 'size_strict': [[0,0],[0,0]],
        'gene_fuzzy':  [[0,0],[0,0]], 'gene_strict': [[0,0],[0,0]],
        'icd_exact':   [[0,0],[0,0]],
    }
    stats_mcnemar = {
        'size_fuzzy':  [[0,0],[0,0]],
        'size_strict': [[0,0],[0,0]],
        'gene_fuzzy':  [[0,0],[0,0]],
        'gene_strict': [[0,0],[0,0]],
        'icd_exact':   [[0,0],[0,0]],
    }
    
    # ── 1. SIZE ──────────────────────────────────────────────────────────
    with open(m0_size_path) as f0, open(m1_size_path) as f1, open(m2_size_path) as f2:
        for l0, l1, l2 in zip(f0, f1, f2):
            r0, r1, r2 = json.loads(l0), json.loads(l1), json.loads(l2)
            split = r1['split']
            target = r1.get('target_text', '')
            diams = get_diams(target)
            
            m0_fz, m0_st = check_size(r0.get('generations', []), diams)
            m1_fz, m1_st = check_size(r1.get('generations', []), diams)
            m2_fz, m2_st = check_size(r2.get('generations', []), diams)
            
            stats_m1['size_fuzzy'][split][0]  += int(m1_fz)
            stats_m1['size_fuzzy'][split][1]  += 1
            stats_m1['size_strict'][split][0] += int(m1_st)
            stats_m1['size_strict'][split][1] += 1
            
            if split == 'train':
                stats_m0_train['size_fuzzy'][0]  += int(m0_fz)
                stats_m0_train['size_fuzzy'][1]  += 1
                stats_m0_train['size_strict'][0] += int(m0_st)
                stats_m0_train['size_strict'][1] += 1
                stats_mcnemar_m0_m1['size_fuzzy'][not m0_fz][not m1_fz] += 1
                stats_mcnemar_m0_m1['size_strict'][not m0_st][not m1_st] += 1
                stats_mcnemar['size_fuzzy'][not m1_fz][not m2_fz] += 1
                stats_mcnemar['size_strict'][not m1_st][not m2_st] += 1

    # ── 2. GENE ──────────────────────────────────────────────────────────
    with open(m0_gene_path) as f0, open(m1_gene_path) as f1, open(m2_gene_path) as f2:
        for l0, l1, l2 in zip(f0, f1, f2):
            r0, r1, r2 = json.loads(l0), json.loads(l1), json.loads(l2)
            split = r1['split']
            
            target = r1.get('target_text', '')
            try: v1 = re.search(r'Pathogenic variant:\s*(.+)', target).group(1).strip()
            except: v1 = 'None identified'
            try: v2 = re.search(r'VUS:\s*(.+)', target).group(1).strip()
            except: v2 = 'None identified'
            
            gene = v1 if v1.lower() not in ['none identified', 'present'] else v2
            if gene.lower() in ['none identified', 'present']: continue
            
            m0_fz, m0_st = check_gene(r0.get('generations', []), gene)
            m1_fz, m1_st = check_gene(r1.get('generations', []), gene)
            m2_fz, m2_st = check_gene(r2.get('generations', []), gene)
            
            stats_m1['gene_fuzzy'][split][0]  += int(m1_fz)
            stats_m1['gene_fuzzy'][split][1]  += 1
            stats_m1['gene_strict'][split][0] += int(m1_st)
            stats_m1['gene_strict'][split][1]  += 1
            
            if split == 'train':
                stats_m0_train['gene_fuzzy'][0]  += int(m0_fz)
                stats_m0_train['gene_fuzzy'][1]  += 1
                stats_m0_train['gene_strict'][0] += int(m0_st)
                stats_m0_train['gene_strict'][1] += 1
                stats_mcnemar_m0_m1['gene_fuzzy'][not m0_fz][not m1_fz] += 1
                stats_mcnemar_m0_m1['gene_strict'][not m0_st][not m1_st] += 1
                stats_mcnemar['gene_fuzzy'][not m1_fz][not m2_fz] += 1
                stats_mcnemar['gene_strict'][not m1_st][not m2_st] += 1

    # ── 3. ICD10 ──────────────────────────────────────────────────────────
    # Denominator: patients with >=3 ground-truth codes only.
    with open(m0_icd_path) as f0, open(m1_icd_path) as f1, open(m2_icd_path) as f2:
        for l0, l1, l2 in zip(f0, f1, f2):
            r0, r1, r2 = json.loads(l0), json.loads(l1), json.loads(l2)
            split = r1['split']
            raw = r1.get('target_icd10', '') or r1.get('target_icd10_raw', '')
            gt_codes = _normalize_icd10_array(raw)
            if len(gt_codes) < 3: continue
            
            m0_ex = check_icd(r0.get('generations', []), gt_codes)
            m1_ex = check_icd(r1.get('generations', []), gt_codes)
            m2_ex = check_icd(r2.get('generations', []), gt_codes)
            
            stats_m1['icd_exact'][split][0] += int(m1_ex)
            stats_m1['icd_exact'][split][1] += 1
            
            if split == 'train':
                stats_m0_train['icd_exact'][0] += int(m0_ex)
                stats_m0_train['icd_exact'][1] += 1
                stats_mcnemar_m0_m1['icd_exact'][not m0_ex][not m1_ex] += 1
                stats_mcnemar['icd_exact'][not m1_ex][not m2_ex] += 1

    # ── Print Output ──────────────────────────────────────────────────────
    print("=== TABLE 1: Train vs Test Extraction (M1 Exact Model) ===")
    for k in ['size_fuzzy', 'size_strict', 'gene_strict', 'gene_fuzzy', 'icd_exact']:
        tr_h, tr_n = stats_m1[k]['train']
        te_h, te_n = stats_m1[k]['test']
        p = calculate_fisher_one_sided(tr_h, tr_n, te_h, te_n)
        tr_pct = tr_h/tr_n*100 if tr_n else 0
        te_pct = te_h/te_n*100 if te_n else 0
        print(f"{k.ljust(15)} | Train: {str(tr_h).rjust(3)}/{str(tr_n).ljust(3)} ({tr_pct:4.1f}%) | "
              f"Test: {str(te_h).rjust(3)}/{str(te_n).ljust(3)} ({te_pct:4.1f}%) | p={p:.4f} {'*' if p<0.05 else ''}")

    print("\n=== TABLE 2: M0 Baseline vs M1 Exact vs M2 Coarsened (Train Cohort Only) ===")
    for k in ['size_fuzzy', 'size_strict', 'gene_strict', 'gene_fuzzy', 'icd_exact']:
        # M0 rate
        m0_h, m0_n = stats_m0_train[k]
        m0_pct = m0_h/m0_n*100 if m0_n else 0
        
        # M1 rate from McNemar table
        table_m1m2 = stats_mcnemar[k]
        m1_succ = table_m1m2[0][0] + table_m1m2[0][1]
        m2_succ = table_m1m2[0][0] + table_m1m2[1][0]
        n_mcn = sum(sum(r) for r in table_m1m2)
        m1_pct = m1_succ/n_mcn*100 if n_mcn else 0
        m2_pct = m2_succ/n_mcn*100 if n_mcn else 0
        
        p_m0_m1 = mcnemar(stats_mcnemar_m0_m1[k], exact=True).pvalue
        p_m1_m2 = mcnemar(table_m1m2, exact=True).pvalue
        
        print(f"{k.ljust(15)} | N={str(n_mcn).ljust(3)} | "
              f"M0: {str(m0_h).rjust(3)} ({m0_pct:4.1f}%) | "
              f"M1: {str(m1_succ).rjust(3)} ({m1_pct:4.1f}%) | "
              f"M2: {str(m2_succ).rjust(3)} ({m2_pct:4.1f}%) | "
              f"p(M0→M1)={p_m0_m1:.4e} {'*' if p_m0_m1<0.05 else ''} | "
              f"p(M1→M2)={p_m1_m2:.4e} {'*' if p_m1_m2<0.05 else ''}")

if __name__ == '__main__':
    evaluate_models()
