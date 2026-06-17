#!/usr/bin/env python3
"""
test_fwd_rev.py — Testa 2 ideias extraídas do potato_biotic_stress.py original:

  F. fwd_rev  : k-mers Forward + Reverse da string (NÃO complemento reverso)
                Como no código original do Colab. Captura simetria de stem-loops.
  G. pca_var  : PCA adaptativo (95% variância) ao invés de 50 componentes fixos
  H. fwd_rev + pca_var combinados
  I. ratio_1_1: NEG_RATIO=1 (1 negativo por positivo, sem SMOTE) — mais simples

Compara com baseline A (v1 pipeline: forward k-mers, PCA-50, SMOTE, NEG_RATIO=3).
"""

import glob, random, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

BASE_PATH = Path('/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data')
FASTA_PATH = BASE_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SIG_BASE   = BASE_PATH / 'Biotic stress/significant_results'
OUT_DIR    = Path('experimento_abordagens')
OUT_DIR.mkdir(exist_ok=True)

K_VALUES  = [2, 3, 4, 5]
N_SPLITS  = 5
N_SEL     = 100
N_PCA     = 50        # v1 fixo
VAR_PCA   = 0.95      # adaptativo (abordagem G)
NEG_RATIO = 3         # v1
NEG_RATIO_1_1 = 1     # nova abordagem I

PATHOGENS = [
    'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
    'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
    'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
    'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
]

# ── Dados ─────────────────────────────────────────────────────────────────────

def load_all_lncrnas():
    seqs = {}
    for rec in SeqIO.parse(str(FASTA_PATH), 'fasta'):
        seqs[rec.id] = str(rec.seq).upper().replace('U', 'T')
    return seqs

def load_pathogen_ids(pathogen):
    csvs = glob.glob(str(SIG_BASE / pathogen / 'lncRNA/Differential expression analysis/*.csv'))
    if not csvs: return set(), set()
    df = pd.read_csv(csvs[0])
    all_de = set(df['lncRNA_transcript_ID'].astype(str))
    pos    = set(df.loc[df['log2FoldChange'] > 1, 'lncRNA_transcript_ID'].astype(str))
    return pos, all_de

def build_dataset(pathogen, all_lncrnas, neg_ratio=NEG_RATIO, seed=42):
    pos_ids, all_de = load_pathogen_ids(pathogen)
    pos_valid = [p for p in pos_ids if p in all_lncrnas]
    neg_pool  = [s for s in all_lncrnas if s not in all_de]
    random.seed(seed)
    n_neg = min(len(neg_pool), len(pos_valid) * neg_ratio)
    neg_ids = random.sample(neg_pool, n_neg)
    return [all_lncrnas[p] for p in pos_valid], [all_lncrnas[n] for n in neg_ids]

# ── Features ──────────────────────────────────────────────────────────────────

def count_kmers_forward(seq, k):
    """K-mers apenas na direção forward (v1)."""
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    total = len(kmers)
    if total == 0: return {}
    c = Counter(kmers)
    return {km: cnt/total for km, cnt in c.items()}

def count_kmers_fwd_rev(seq, k):
    """
    K-mers Forward + Reverse da string (como no potato_biotic_stress.py original).
    ATENÇÃO: 'reverse' aqui é a string invertida, NÃO o complemento reverso.
    Ex: ATCG → reverse = GCTA (não CGAT = complemento reverso)
    """
    fwd = [seq[i:i+k] for i in range(len(seq)-k+1)]
    rev = [seq[::-1][i:i+k] for i in range(len(seq[::-1])-k+1)]
    kmers = fwd + rev
    total = len(kmers)
    if total == 0: return {}
    c = Counter(kmers)
    return {km: cnt/total for km, cnt in c.items()}

def build_matrix(sequences, k, use_fwd_rev=False, kmer_dict=None):
    fn = count_kmers_fwd_rev if use_fwd_rev else count_kmers_forward
    if kmer_dict is None:
        vocab = set()
        for seq in sequences:
            vocab.update(fn(seq, k).keys())
        kmer_dict = {km: i for i, km in enumerate(sorted(vocab))}
    sorted_km = sorted(kmer_dict)
    rows = []
    for seq in sequences:
        d = fn(seq, k)
        rows.append([d.get(km, 0.0) for km in sorted_km])
    return np.array(rows, dtype=np.float32), kmer_dict

# ── Pré-processamento ─────────────────────────────────────────────────────────

def fit_preprocess(X_tr, y_tr, X_te=None, adaptive_pca=False):
    k_sel    = min(N_SEL, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k_sel).fit(X_tr, y_tr)
    X_tr_s   = selector.transform(X_tr)
    X_te_s   = selector.transform(X_te) if X_te is not None else None

    if adaptive_pca:
        n_max = min(X_tr_s.shape[0]-1, X_tr_s.shape[1])
        pca   = PCA(n_components=min(VAR_PCA, n_max)).fit(X_tr_s)
    else:
        n_pca = min(N_PCA, X_tr_s.shape[1], X_tr_s.shape[0]-1)
        pca   = PCA(n_components=n_pca).fit(X_tr_s)

    X_tr_p = pca.transform(X_tr_s)
    X_te_p = pca.transform(X_te_s) if X_te_s is not None else None
    scaler = StandardScaler().fit(X_tr_p)
    return scaler.transform(X_tr_p), scaler.transform(X_te_p) if X_te_p is not None else None

def apply_smote(X, y):
    n_pos = int(np.sum(y == 1))
    k_nb  = min(3, n_pos-1)
    if k_nb < 1: return X, y
    try:
        return SMOTE(random_state=42, k_neighbors=k_nb).fit_resample(X, y)
    except:
        return X, y

def make_classifiers():
    return {
        'GaussianNB':         GaussianNB(),
        'SVC_RBF':            SVC(kernel='rbf', probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'NuSVC':              NuSVC(kernel='rbf', probability=True, random_state=42),
        'KNN':                KNeighborsClassifier(n_neighbors=5),
    }

# ── CV ────────────────────────────────────────────────────────────────────────

def cv_evaluate(X, y, clf_name, adaptive_pca=False, use_smote=True, n_splits=N_SPLITS):
    n_pos  = int(np.sum(y == 1))
    actual = min(n_splits, n_pos)
    if actual < 2: return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    f1_list = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        try:
            X_tr_f, X_te_f = fit_preprocess(X_tr, y_tr, X_te, adaptive_pca=adaptive_pca)
        except: continue
        if use_smote:
            X_tr_f, y_tr = apply_smote(X_tr_f, y_tr)
        clf = make_classifiers()[clf_name]
        try:
            clf.fit(X_tr_f, y_tr)
            f1_list.append(f1_score(y_te, clf.predict(X_te_f), zero_division=0))
        except: continue

    if not f1_list: return None
    return {'f1_mean': float(np.mean(f1_list)), 'f1_std': float(np.std(f1_list))}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('=' * 80)
    print('  EXPERIMENTO F-I: Forward+Rev K-mers | PCA Adaptativo | Ratio 1:1')
    print('  F=fwd+rev | G=pca_var | H=fwd+rev+pca_var | I=ratio_1_1(sem SMOTE)')
    print('=' * 80)

    print('\nCarregando sequências...')
    all_lncrnas = load_all_lncrnas()
    print(f'  ✓ {len(all_lncrnas)} lncRNAs')

    v1_best = pd.read_csv('grid_search_results/melhores_modelos.csv').set_index('pathogen')['f1'].to_dict()
    all_rows = []

    for p_idx, pathogen in enumerate(PATHOGENS, 1):
        print(f'\n[{p_idx:02d}/{len(PATHOGENS)}] {pathogen}')
        pos_seqs_3, neg_seqs_3 = build_dataset(pathogen, all_lncrnas, neg_ratio=NEG_RATIO)
        pos_seqs_1, neg_seqs_1 = build_dataset(pathogen, all_lncrnas, neg_ratio=NEG_RATIO_1_1)
        n_pos = len(pos_seqs_3)
        if n_pos < 4:
            print(f'  ⚠ Insuficiente ({n_pos} pos)')
            continue

        all_seqs_3 = pos_seqs_3 + neg_seqs_3
        y3         = np.array([1]*len(pos_seqs_3) + [0]*len(neg_seqs_3))
        all_seqs_1 = pos_seqs_1 + neg_seqs_1
        y1         = np.array([1]*len(pos_seqs_1) + [0]*len(neg_seqs_1))

        v1_ref = v1_best.get(pathogen, 0)
        print(f'  n_pos={n_pos}  neg(3:1)={len(neg_seqs_3)}  neg(1:1)={len(neg_seqs_1)}  v1_ref={v1_ref:.4f}')

        # Pré-computa matrizes
        mats = {}
        for k in K_VALUES:
            Xf3, _ = build_matrix(all_seqs_3, k, use_fwd_rev=False)  # A: baseline forward
            Xr3, _ = build_matrix(all_seqs_3, k, use_fwd_rev=True)   # F/G/H: fwd+rev
            Xr1, _ = build_matrix(all_seqs_1, k, use_fwd_rev=False)  # I: ratio 1:1
            mats[k] = {'fwd3': Xf3, 'fwdrev3': Xr3, 'fwd1': Xr1}

        best = {'A_baseline': 0, 'F_fwdrev': 0, 'G_pca_var': 0, 'H_fwdrev_pca': 0, 'I_ratio11': 0}

        for k in K_VALUES:
            for clf_name in make_classifiers().keys():
                # A: baseline (v1) — forward k-mers, PCA-50, SMOTE, ratio 3:1
                res_a = cv_evaluate(mats[k]['fwd3'],    y3, clf_name, adaptive_pca=False, use_smote=True)
                # F: forward + reverse k-mers, PCA-50, SMOTE, ratio 3:1
                res_f = cv_evaluate(mats[k]['fwdrev3'], y3, clf_name, adaptive_pca=False, use_smote=True)
                # G: forward k-mers, PCA adaptativo 95%, SMOTE, ratio 3:1
                res_g = cv_evaluate(mats[k]['fwd3'],    y3, clf_name, adaptive_pca=True,  use_smote=True)
                # H: forward + reverse, PCA adaptativo 95%, SMOTE, ratio 3:1
                res_h = cv_evaluate(mats[k]['fwdrev3'], y3, clf_name, adaptive_pca=True,  use_smote=True)
                # I: forward k-mers, PCA-50, sem SMOTE, ratio 1:1
                res_i = cv_evaluate(mats[k]['fwd1'],    y1, clf_name, adaptive_pca=False, use_smote=False)

                f = lambda r: f"{r['f1_mean']:.4f}" if r else ' N/A '
                print(f'  K={k} {clf_name:<22} A={f(res_a)} F={f(res_f)} G={f(res_g)} H={f(res_h)} I={f(res_i)}')

                for label, res, y_ref in [('A_baseline', res_a, y3), ('F_fwdrev', res_f, y3),
                                          ('G_pca_var', res_g, y3), ('H_fwdrev_pca', res_h, y3),
                                          ('I_ratio11', res_i, y1)]:
                    if res:
                        f1 = res['f1_mean']
                        if f1 > best[label]: best[label] = f1
                        all_rows.append({'pathogen': pathogen, 'approach': label, 'k': k,
                                         'classifier': clf_name, 'f1_mean': round(f1, 4),
                                         'f1_std': round(res['f1_std'], 4), 'n_pos': n_pos})

        print(f'\n  === Resumo {pathogen} | v1_ref={v1_ref:.4f} ===')
        for ap, f1 in best.items():
            delta = f1 - v1_ref
            mark  = '★' if f1 == max(best.values()) else ' '
            print(f'  {mark} {ap:<15} F1={f1:.4f}  Δ={delta:+.4f}')

        pd.DataFrame(all_rows).to_csv(str(OUT_DIR / 'resultados_fwd_rev_parciais.csv'), index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(str(OUT_DIR / 'resultados_fwd_rev_completos.csv'), index=False)

    print('\n' + '='*80)
    print('  RESUMO FINAL — Média F1 sobre todos os patógenos')
    print('='*80)
    for ap in ['A_baseline', 'F_fwdrev', 'G_pca_var', 'H_fwdrev_pca', 'I_ratio11']:
        sub = df[df['approach'] == ap]
        bp  = sub.groupby('pathogen')['f1_mean'].max()
        mean_f1 = bp.mean()
        n70  = (bp >= 0.70).sum()
        nv1  = sum(bp.get(p, 0) > v1_best.get(p, 0) for p in PATHOGENS)
        print(f'  {ap:<18}  F1 médio={mean_f1:.4f}  F1≥0.70: {n70}/12  bate v1: {nv1}/12')

    from datetime import timedelta
    print(f'\n  Tempo total: {timedelta(seconds=int(time.time()-t0))}')
    print(f'  Resultados: {OUT_DIR}/')

if __name__ == '__main__':
    main()
