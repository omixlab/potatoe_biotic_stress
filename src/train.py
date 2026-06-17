#!/usr/bin/env python3
"""
retrain_v4.py — Retreino com pipeline v4: NEG_RATIO=1, sem SMOTE.

Descoberta experimental (test_fwd_rev.py):
  NEG_RATIO=1 (1 negativo por positivo, classes já balanceadas) sem SMOTE
  melhora F1 em +0.16..+0.20 sobre v1 em TODOS os 12 patógenos testados.

  Por quê funciona:
  - v1 usava NEG_RATIO=3 + SMOTE → criava ~2x positivos sintéticos por interpolação
  - Amostras sintéticas do SMOTE borram a fronteira de decisão do GaussianNB
  - Com 1:1 real, GaussianNB aprende a distribuição positiva verdadeira diretamente

Pipeline v4:
  build_dataset → NEG_RATIO=1, sem SMOTE
  SelectKBest(f_classif, 100) → PCA(50) → StandardScaler → Classifier
  Grid search: K=2..5 × 8 classifiers, 5-fold CV honesto
"""

import glob, json, random, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import timedelta

import joblib
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

BASE_PATH  = Path('/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data')
FASTA_PATH = BASE_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SIG_BASE   = BASE_PATH / 'Biotic stress/significant_results'
MODEL_DIR  = Path('models/saved')
PREP_DIR   = MODEL_DIR
RESULT_DIR = Path('results/v4')

for d in [MODEL_DIR, PREP_DIR, RESULT_DIR]:
    d.mkdir(exist_ok=True)

K_VALUES   = [2, 3, 4, 5, 6]
N_SPLITS   = 5
N_SEL      = 100
N_PCA      = 50
NEG_RATIO  = 1      # ← chave: 1:1 sem SMOTE

PATHOGENS = [
    'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
    'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
    'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
    'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
]

# ─────────────────────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────────────────────

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

def build_dataset(pathogen, all_lncrnas, seed=42):
    pos_ids, all_de_ids = load_pathogen_ids(pathogen)
    pos_valid = [p for p in pos_ids if p in all_lncrnas]
    neg_pool  = [s for s in all_lncrnas if s not in all_de_ids]
    random.seed(seed)
    n_neg   = min(len(neg_pool), len(pos_valid) * NEG_RATIO)
    neg_ids = random.sample(neg_pool, n_neg)
    return [all_lncrnas[p] for p in pos_valid], [all_lncrnas[n] for n in neg_ids]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def count_kmers(seq, k):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    total = len(kmers)
    if total == 0: return {}
    c = Counter(kmers)
    return {km: cnt/total for km, cnt in c.items()}

def build_feature_matrix(sequences, k, kmer_dict=None):
    if kmer_dict is None:
        vocab = set()
        for seq in sequences:
            for i in range(len(seq)-k+1): vocab.add(seq[i:i+k])
        kmer_dict = {km: i for i, km in enumerate(sorted(vocab))}
    sorted_km = sorted(kmer_dict)
    rows = []
    for seq in sequences:
        d = count_kmers(seq, k)
        rows.append([d.get(km, 0.0) for km in sorted_km])
    return np.array(rows, dtype=np.float32), kmer_dict

# ─────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (sem SMOTE)
# ─────────────────────────────────────────────────────────────────────────────

def fit_preprocess(X_tr, y_tr, X_te=None):
    k_sel    = min(N_SEL, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k_sel).fit(X_tr, y_tr)
    X_tr_s   = selector.transform(X_tr)
    X_te_s   = selector.transform(X_te) if X_te is not None else None
    n_pca    = min(N_PCA, X_tr_s.shape[1], X_tr_s.shape[0]-1)
    pca      = PCA(n_components=n_pca).fit(X_tr_s)
    X_tr_p   = pca.transform(X_tr_s)
    X_te_p   = pca.transform(X_te_s) if X_te_s is not None else None
    scaler   = StandardScaler().fit(X_tr_p)
    return (scaler.transform(X_tr_p),
            scaler.transform(X_te_p) if X_te_p is not None else None,
            selector, pca, scaler)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def make_classifiers():
    return {
        'GaussianNB':         GaussianNB(),
        'BernoulliNB':        BernoulliNB(),
        'QDA':                QuadraticDiscriminantAnalysis(),
        'SVC_RBF':            SVC(kernel='rbf', probability=True, random_state=42),
        'NuSVC':              NuSVC(kernel='rbf', probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN':                KNeighborsClassifier(n_neighbors=5),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'MLP':                MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }

# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH COM 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────

def cv_evaluate(X, y, clf_name, n_splits=N_SPLITS):
    n_pos  = int(np.sum(y == 1))
    actual = min(n_splits, n_pos)
    if actual < 2: return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    f1_test_list  = []
    f1_train_list = []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        try:
            X_tr_f, X_te_f, *_ = fit_preprocess(X_tr, y_tr, X_te)
        except: continue
        clf = make_classifiers()[clf_name]
        try:
            clf.fit(X_tr_f, y_tr)
            f1_test_list.append(f1_score(y_te,  clf.predict(X_te_f),  zero_division=0))
            f1_train_list.append(f1_score(y_tr, clf.predict(X_tr_f), zero_division=0))
        except: continue

    if not f1_test_list: return None
    return (float(np.mean(f1_test_list)),  float(np.std(f1_test_list)),
            float(np.mean(f1_train_list)), float(np.std(f1_train_list)))

def grid_search_pathogen(pathogen, all_lncrnas):
    pos_seqs, neg_seqs = build_dataset(pathogen, all_lncrnas)
    n_pos, n_neg = len(pos_seqs), len(neg_seqs)
    if n_pos < 4:
        print(f'  ⚠ Insuficiente ({n_pos} pos)')
        return None

    all_seqs = pos_seqs + neg_seqs
    y        = np.array([1]*n_pos + [0]*n_neg)
    print(f'  Dataset: {n_pos} pos | {n_neg} neg (ratio {n_neg/n_pos:.1f}:1)')

    results = []
    for k in K_VALUES:
        X, kd = build_feature_matrix(all_seqs, k)
        for clf_name in make_classifiers():
            res = cv_evaluate(X, y, clf_name)
            if res:
                f1_mean, f1_std, f1_tr, f1_tr_std = res
                gap = f1_tr - f1_mean
                results.append({
                    'pathogen': pathogen, 'k': k, 'model_type': clf_name,
                    'f1': round(f1_mean, 4), 'f1_std': round(f1_std, 4),
                    'f1_train': round(f1_tr, 4), 'gap': round(gap, 4),
                    'n_pos': n_pos, 'n_neg': n_neg,
                })
                diag = 'OVERFIT' if gap > 0.15 else ('UNDERFIT' if f1_mean < 0.55 and gap < 0.05 else '')
                flag = f'  ← {diag}' if diag else ''
                print(f'  K={k} {clf_name:<22} CV={f1_mean:.4f}±{f1_std:.4f}  TR={f1_tr:.4f}  gap={gap:+.4f}{flag}')

    if not results: return None
    df = pd.DataFrame(results)
    # Penaliza modelos com gap > 0.15 para evitar selecionar overfitters
    df['f1_adj'] = df['f1'] - df['gap'].clip(lower=0.15).sub(0.15).mul(0.5)
    best = df.loc[df['f1_adj'].idxmax()].to_dict()
    gap_best = best['gap']
    diag = 'OVERFIT' if gap_best > 0.15 else ('UNDERFIT' if best['f1'] < 0.55 and gap_best < 0.05 else 'OK')
    print(f'  ★ Melhor: K={best["k"]} {best["model_type"]} CV={best["f1"]:.4f}  TR={best["f1_train"]:.4f}  gap={gap_best:+.4f}  [{diag}]')
    return df, best

# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN NO DATASET COMPLETO COM MELHOR (K, CLF)
# ─────────────────────────────────────────────────────────────────────────────

def retrain_full(pathogen, k, clf_name, all_lncrnas):
    pos_seqs, neg_seqs = build_dataset(pathogen, all_lncrnas)
    all_seqs = pos_seqs + neg_seqs
    y        = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))

    X, kd = build_feature_matrix(all_seqs, k)
    X_f, _, selector, pca, scaler = fit_preprocess(X, y)

    raw_clf = make_classifiers()[clf_name]
    n_cv = min(3, int(np.sum(y == 1)) // 2)
    clf = CalibratedClassifierCV(raw_clf, method='sigmoid', cv=max(2, n_cv))
    clf.fit(X_f, y)

    name = pathogen.replace(' ', '_')
    joblib.dump(clf,      str(MODEL_DIR / f'{name}_model.pkl'))
    joblib.dump(selector, str(MODEL_DIR / f'{name}_selector.pkl'))
    joblib.dump(pca,      str(MODEL_DIR / f'{name}_pca.pkl'))
    joblib.dump(scaler,   str(MODEL_DIR / f'{name}_scaler.pkl'))
    joblib.dump(kd,       str(MODEL_DIR / f'{name}_kmer_dict.pkl'))
    # v4 não tem PCA removível — salva flag para app saber
    return kd

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('=' * 80)
    print('  RETRAIN v4 — NEG_RATIO=1, sem SMOTE')
    print('  Pipeline: SelectKBest(100) → PCA(50) → StandardScaler → Classifier')
    print('=' * 80)

    print('\nCarregando sequências...')
    all_lncrnas = load_all_lncrnas()
    print(f'  ✓ {len(all_lncrnas)} lncRNAs\n')

    v1_ref = pd.read_csv('grid_search_results/melhores_modelos.csv').set_index('pathogen')['f1'].to_dict()

    all_results  = []
    best_models  = []
    config       = {}

    for p_idx, pathogen in enumerate(PATHOGENS, 1):
        print(f'[{p_idx:02d}/{len(PATHOGENS)}] {pathogen}')
        out = grid_search_pathogen(pathogen, all_lncrnas)
        if out is None: continue
        df_p, best = out

        all_results.append(df_p)
        best_models.append({
            'pathogen':   pathogen,
            'k':          best['k'],
            'model_type': best['model_type'],
            'f1':         best['f1'],
            'f1_std':     best['f1_std'],
            'f1_train':   best['f1_train'],
            'gap':        best['gap'],
            'n_pos':      best['n_pos'],
            'f1_v1':      round(v1_ref.get(pathogen, 0), 4),
            'delta_v1':   round(best['f1'] - v1_ref.get(pathogen, 0), 4),
        })
        print()

    df_all   = pd.concat(all_results, ignore_index=True)
    df_best  = pd.DataFrame(best_models)
    df_all.to_csv(str(RESULT_DIR / 'grid_search_results.csv'), index=False)
    df_best.to_csv(str(RESULT_DIR / 'melhores_modelos.csv'),  index=False)

    print('\n' + '=' * 80)
    print('  RETREINANDO modelos finais (dataset completo)...')
    print('=' * 80)

    for _, row in df_best.iterrows():
        pathogen  = row['pathogen']
        k         = int(row['k'])
        clf_name  = row['model_type']
        f1        = row['f1']
        print(f'  ► {pathogen:<35} K={k}  {clf_name:<22}  F1={f1:.4f}')
        retrain_full(pathogen, k, clf_name, all_lncrnas)
        config[pathogen] = {'k': k, 'model_type': clf_name, 'f1': float(f1)}
        print(f'    ✓ salvo')

    full_config = {
        '_version':         'v4',
        '_canonical_kmers': False,
        '_pca':             True,
        '_bio_features':    False,
        '_smote':           False,
        '_neg_ratio':       1,
        **config,
    }
    with open('best_models_config.json', 'w') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 80)
    print('  RESULTADO FINAL v4')
    print('=' * 80)
    df_best['diag'] = df_best.apply(
        lambda r: 'OVERFIT' if r['gap'] > 0.15 else ('UNDERFIT' if r['f1'] < 0.55 and r['gap'] < 0.05 else 'OK'), axis=1)
    print(df_best[['pathogen', 'k', 'model_type', 'f1', 'f1_train', 'gap', 'diag', 'f1_v1', 'delta_v1']].to_string(index=False))
    print(f'\n  F1 médio v4: {df_best["f1"].mean():.4f}')
    print(f'  F1 médio v1: {df_best["f1_v1"].mean():.4f}')
    print(f'  Δ médio:     {df_best["delta_v1"].mean():+.4f}')
    print(f'  F1≥0.70:     {(df_best["f1"] >= 0.70).sum()}/12 patógenos')
    print(f'\n  Tempo total: {timedelta(seconds=int(time.time()-t0))}')
    print('\n[✓] best_models_config.json atualizado para v4')
    print('Próximo passo: python3 app_predicao.py')

if __name__ == '__main__':
    main()
