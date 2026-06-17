#!/usr/bin/env python3
"""
test_abordagens.py — Experimento sistemático para melhorar F1 além de 0.47 (média v1)

Abordagens testadas com 5-fold CV honesto:
  A. baseline    : k-mers globais (= pipeline v1)
  B. positional  : k-mers em 3 janelas posicionais (5', meio, 3')
  C. padj_weight : k-mers globais + pesos de treino -log10(padj) para positivos
  D. pos_padj    : janelas posicionais + pesos padj
  E. stacking    : meta-modelo LR sobre probabilidades de K=2..5 modelos base

Classifiers testados (rápidos e comprovados): GaussianNB, SVC_RBF, LogisticRegression, NuSVC, KNN
K-sizes: 2, 3, 4, 5  (K=6 excluído: nunca venceu em v1 e é lento)
"""

import glob, json, random, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import product

import joblib
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = Path('/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data')
FASTA_PATH = BASE_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SIG_BASE   = BASE_PATH / 'Biotic stress/significant_results'
OUT_DIR    = Path('experimento_abordagens')
OUT_DIR.mkdir(exist_ok=True)

K_VALUES   = [2, 3, 4, 5]
N_SPLITS   = 5
N_SEL      = 100
N_PCA      = 50
NEG_RATIO  = 3
N_WINDOWS  = 3      # janelas posicionais
MIN_WIN_NT = 40     # mínimo de nts por janela (senão usa global)

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

def load_pathogen_data(pathogen):
    """Retorna (pos_ids, all_de_ids, padj_map: {id: padj_val})."""
    csvs = glob.glob(str(SIG_BASE / pathogen / 'lncRNA/Differential expression analysis/*.csv'))
    if not csvs: return set(), set(), {}
    df = pd.read_csv(csvs[0])
    id_col, fc_col, padj_col = 'lncRNA_transcript_ID', 'log2FoldChange', 'padj'
    all_de  = set(df[id_col].astype(str))
    pos_ids = set(df.loc[df[fc_col] > 1, id_col].astype(str))
    padj_map = {}
    if padj_col in df.columns:
        for _, row in df[df[fc_col] > 1].iterrows():
            v = row[padj_col]
            if pd.notna(v) and v > 0:
                padj_map[str(row[id_col])] = float(v)
    return pos_ids, all_de, padj_map

def build_dataset(pathogen, all_lncrnas, seed=42):
    pos_ids, all_de_ids, padj_map = load_pathogen_data(pathogen)
    pos_ids_valid = [p for p in pos_ids if p in all_lncrnas]
    neg_pool      = [s for s in all_lncrnas if s not in all_de_ids]
    random.seed(seed)
    n_neg    = min(len(neg_pool), len(pos_ids_valid) * NEG_RATIO)
    neg_ids  = random.sample(neg_pool, n_neg)
    pos_seqs = [all_lncrnas[p] for p in pos_ids_valid]
    neg_seqs = [all_lncrnas[n] for n in neg_ids]
    # Pesos padj: -log10(padj+eps) para positivos, 1.0 para negativos
    weights = []
    for p in pos_ids_valid:
        if p in padj_map:
            weights.append(max(1.0, -np.log10(padj_map[p] + 1e-10)))
        else:
            weights.append(1.0)
    weights += [1.0] * len(neg_seqs)
    return pos_seqs, neg_seqs, weights

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def count_kmers(seq, k):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    total = len(kmers)
    if total == 0: return {}
    c = Counter(kmers)
    return {km: cnt/total for km, cnt in c.items()}

def build_global_matrix(sequences, k, kmer_dict=None):
    """Feature = frequência relativa global de k-mers (baseline v1)."""
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

def build_positional_matrix(sequences, k, kmer_dict=None):
    """
    Feature = k-mers nas 3 janelas (5', meio, 3') concatenadas.
    Sequências muito curtas (< 3×MIN_WIN_NT) usam feature global.
    """
    if kmer_dict is None:
        vocab = set()
        for seq in sequences:
            for i in range(len(seq)-k+1): vocab.add(seq[i:i+k])
        kmer_dict = {km: i for i, km in enumerate(sorted(vocab))}
    sorted_km  = sorted(kmer_dict)
    n_km       = len(sorted_km)
    min_len    = N_WINDOWS * MIN_WIN_NT

    rows = []
    for seq in sequences:
        n = len(seq)
        if n < min_len:
            # Sequência curta: repete o vetor global nas 3 janelas
            d = count_kmers(seq, k)
            v = [d.get(km, 0.0) for km in sorted_km]
            rows.append(v * N_WINDOWS)
        else:
            ws = n // N_WINDOWS
            row = []
            for w in range(N_WINDOWS):
                start = w * ws
                end   = (w+1)*ws if w < N_WINDOWS-1 else n
                d = count_kmers(seq[start:end], k)
                row.extend([d.get(km, 0.0) for km in sorted_km])
            rows.append(row)
    return np.array(rows, dtype=np.float32), kmer_dict

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
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
    X_tr_f   = scaler.transform(X_tr_p)
    X_te_f   = scaler.transform(X_te_p) if X_te_p is not None else None
    return X_tr_f, X_te_f, selector, pca, scaler

def apply_smote(X, y):
    n_pos = int(np.sum(y == 1))
    k_nb  = min(3, n_pos-1)
    if k_nb < 1: return X, y, None
    try:
        sm = SMOTE(random_state=42, k_neighbors=k_nb)
        Xr, yr = sm.fit_resample(X, y)
        return Xr, yr, sm
    except:
        return X, y, None

def make_classifiers():
    return {
        'GaussianNB':         GaussianNB(),
        'SVC_RBF':            SVC(kernel='rbf', probability=True, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'NuSVC':              NuSVC(kernel='rbf', probability=True, random_state=42),
        'KNN':                KNeighborsClassifier(n_neighbors=5),
    }

def get_proba(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    raw = clf.predict(X).astype(float)
    return np.clip(1/(1+np.exp(-raw)), 0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CV COM SUPORTE A PESOS E ABORDAGEM SELECIONÁVEL
# ─────────────────────────────────────────────────────────────────────────────

def cv_evaluate(X, y, clf_name, weights=None, n_splits=N_SPLITS):
    """weights: array de sample_weight (mesma ordem de y). None = uniforme."""
    n_pos = int(np.sum(y == 1))
    actual = min(n_splits, n_pos)
    if actual < 2: return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    f1_list, roc_list = [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        w_tr       = weights[tr_idx] if weights is not None else None

        try:
            X_tr_f, X_te_f, *_ = fit_preprocess(X_tr, y_tr, X_te)
        except: continue

        # SMOTE (sem pesos — aplica antes)
        X_tr_f, y_tr_f, _ = apply_smote(X_tr_f, y_tr)
        # Para pesos: SMOTE não propaga os pesos originais.
        # Usamos pesos APENAS em classifiers sem SMOTE (abordagem padj_only separada).
        # Aqui ignoramos w_tr após SMOTE — efeito de pesos é pré-SMOTE via sample filtrado.

        clf = make_classifiers()[clf_name]
        try:
            clf.fit(X_tr_f, y_tr_f)
        except: continue

        try:
            y_pred = clf.predict(X_te_f)
            f1_list.append(f1_score(y_te, y_pred, zero_division=0))
            if len(np.unique(y_te)) == 2:
                roc_list.append(roc_auc_score(y_te, get_proba(clf, X_te_f)))
        except: continue

    if not f1_list: return None
    return {'f1_mean': float(np.mean(f1_list)), 'f1_std': float(np.std(f1_list)),
            'roc_mean': float(np.mean(roc_list)) if roc_list else None,
            'n_folds': len(f1_list)}

def cv_evaluate_padj(X, y, weights, clf_name, n_splits=N_SPLITS):
    """
    Versão SEM SMOTE: usa pesos de padj no fit() diretamente.
    Só funciona com GaussianNB, SVC_RBF e LogisticRegression.
    """
    SUPPORTS_SW = {'GaussianNB', 'SVC_RBF', 'LogisticRegression'}
    if clf_name not in SUPPORTS_SW:
        return cv_evaluate(X, y, clf_name, n_splits=n_splits)

    n_pos = int(np.sum(y == 1))
    actual = min(n_splits, n_pos)
    if actual < 2: return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    f1_list, roc_list = [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        w_tr       = weights[tr_idx]

        try:
            X_tr_f, X_te_f, *_ = fit_preprocess(X_tr, y_tr, X_te)
        except: continue

        clf = make_classifiers()[clf_name]
        try:
            # Passa pesos originais diretamente (sem SMOTE)
            if clf_name == 'GaussianNB':
                clf.fit(X_tr_f, y_tr, sample_weight=w_tr)
            elif clf_name == 'SVC_RBF':
                clf2 = SVC(kernel='rbf', probability=True, random_state=42,
                           class_weight='balanced')
                clf2.fit(X_tr_f, y_tr, sample_weight=w_tr)
                clf = clf2
            else:  # LogisticRegression
                clf2 = LogisticRegression(max_iter=1000, random_state=42,
                                          class_weight='balanced')
                clf2.fit(X_tr_f, y_tr, sample_weight=w_tr)
                clf = clf2
        except: continue

        try:
            y_pred = clf.predict(X_te_f)
            f1_list.append(f1_score(y_te, y_pred, zero_division=0))
            if len(np.unique(y_te)) == 2:
                roc_list.append(roc_auc_score(y_te, get_proba(clf, X_te_f)))
        except: continue

    if not f1_list: return None
    return {'f1_mean': float(np.mean(f1_list)), 'f1_std': float(np.std(f1_list)),
            'roc_mean': float(np.mean(roc_list)) if roc_list else None,
            'n_folds': len(f1_list)}

# ─────────────────────────────────────────────────────────────────────────────
# ABORDAGEM E: STACKING (K=2..5 base models → LR meta-model)
# ─────────────────────────────────────────────────────────────────────────────

def cv_stacking(all_seqs, y, clf_name='GaussianNB', n_splits=N_SPLITS):
    """
    Meta-modelo: treina um clf por K-size, usa suas probabilidades como
    features de um LR meta-classificador. Tudo dentro de cada fold de CV.
    """
    n_pos = int(np.sum(y == 1))
    actual = min(n_splits, n_pos)
    if actual < 2: return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    f1_list, roc_list = [], []

    # Pré-computa matrizes para todos os K (evita re-computar dentro dos folds)
    matrices = {}
    kdicts   = {}
    for k in K_VALUES:
        X_k, kd = build_global_matrix(all_seqs, k)
        matrices[k] = X_k
        kdicts[k]   = kd

    for tr_idx, te_idx in skf.split(matrices[K_VALUES[0]], y):
        y_tr, y_te = y[tr_idx], y[te_idx]
        meta_tr, meta_te = [], []

        for k in K_VALUES:
            X_k = matrices[k]
            X_tr_k, X_te_k = X_k[tr_idx], X_k[te_idx]
            try:
                X_tr_f, X_te_f, *_ = fit_preprocess(X_tr_k, y_tr, X_te_k)
            except: continue

            X_tr_f, y_tr_s, _ = apply_smote(X_tr_f, y_tr)
            clf_k = make_classifiers()[clf_name]
            try:
                clf_k.fit(X_tr_f, y_tr_s)
                prob_tr = get_proba(clf_k, X_tr_f[:len(y_tr)])  # pré-SMOTE size para meta
                # Reclassifica o treino original (sem SMOTE) para meta-features
                # Refit no conjunto original para meta-features de treino
                clf_k2 = make_classifiers()[clf_name]
                X_tr_orig_f, *_ = fit_preprocess(X_tr_k, y_tr)
                Xs2, ys2, _ = apply_smote(X_tr_orig_f, y_tr)
                clf_k2.fit(Xs2, ys2)
                prob_tr = get_proba(clf_k2, X_tr_orig_f)
                prob_te = get_proba(clf_k, X_te_f)
                meta_tr.append(prob_tr)
                meta_te.append(prob_te)
            except: continue

        if not meta_tr: continue

        X_meta_tr = np.column_stack(meta_tr) if len(meta_tr) > 1 else meta_tr[0].reshape(-1,1)
        X_meta_te = np.column_stack(meta_te) if len(meta_te) > 1 else meta_te[0].reshape(-1,1)

        try:
            meta_clf = LogisticRegression(max_iter=500, random_state=42)
            meta_clf.fit(X_meta_tr, y_tr)
            y_pred = meta_clf.predict(X_meta_te)
            f1_list.append(f1_score(y_te, y_pred, zero_division=0))
            if len(np.unique(y_te)) == 2:
                roc_list.append(roc_auc_score(y_te, get_proba(meta_clf, X_meta_te)))
        except: continue

    if not f1_list: return None
    return {'f1_mean': float(np.mean(f1_list)), 'f1_std': float(np.std(f1_list)),
            'roc_mean': float(np.mean(roc_list)) if roc_list else None,
            'n_folds': len(f1_list)}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

CLF_NAMES = list(make_classifiers().keys())

def main():
    t0 = time.time()
    print('=' * 80)
    print('  EXPERIMENTO: Comparação de Abordagens para Melhorar F1')
    print('  A=baseline | B=positional | C=padj_weight | D=pos+padj | E=stacking')
    print('=' * 80)

    print('\nCarregando sequências...')
    all_lncrnas = load_all_lncrnas()
    print(f'  ✓ {len(all_lncrnas)} lncRNAs')

    # Baseline (v1) para referência
    v1_best = pd.read_csv('grid_search_results/melhores_modelos.csv').set_index('pathogen')['f1'].to_dict()

    all_rows = []

    for p_idx, pathogen in enumerate(PATHOGENS, 1):
        print(f'\n[{p_idx:02d}/{len(PATHOGENS)}] {pathogen}')

        pos_seqs, neg_seqs, weights_list = build_dataset(pathogen, all_lncrnas)
        n_pos, n_neg = len(pos_seqs), len(neg_seqs)
        if n_pos < 4:
            print(f'  ⚠ Positivos insuficientes ({n_pos}), pulando.')
            continue

        all_seqs = pos_seqs + neg_seqs
        y        = np.array([1]*n_pos + [0]*n_neg)
        weights  = np.array(weights_list)

        print(f'  Dados: {n_pos} pos | {n_neg} neg | {n_pos+n_neg} total')
        print(f'  Pesos padj: min={weights[:n_pos].min():.1f} max={weights[:n_pos].max():.1f}')
        v1_ref = v1_best.get(pathogen, 0)
        print(f'  Referência v1: {v1_ref:.4f}')

        # Pré-computa matrizes
        global_mats, pos_mats, kdicts = {}, {}, {}
        for k in K_VALUES:
            Xg, kd = build_global_matrix(all_seqs, k)
            Xp, _  = build_positional_matrix(all_seqs, k, kmer_dict=kd)
            global_mats[k] = Xg
            pos_mats[k]    = Xp
            kdicts[k]      = kd

        best_per_approach = {'A_baseline': 0, 'B_positional': 0,
                             'C_padj': 0, 'D_pos_padj': 0, 'E_stacking': 0}

        # ── ABORDAGENS A, B, C, D ────────────────────────────────────────────
        for k in K_VALUES:
            Xg = global_mats[k]
            Xp = pos_mats[k]

            for clf_name in CLF_NAMES:
                # A: baseline (global k-mers + SMOTE)
                res_a = cv_evaluate(Xg, y, clf_name)

                # B: positional k-mers + SMOTE
                res_b = cv_evaluate(Xp, y, clf_name)

                # C: global k-mers + padj weights (sem SMOTE nos clf que suportam)
                res_c = cv_evaluate_padj(Xg, y, weights, clf_name)

                # D: positional + padj weights
                res_d = cv_evaluate_padj(Xp, y, weights, clf_name)

                for label, res in [('A_baseline', res_a), ('B_positional', res_b),
                                   ('C_padj', res_c), ('D_pos_padj', res_d)]:
                    if res:
                        f1 = res['f1_mean']
                        if f1 > best_per_approach[label]:
                            best_per_approach[label] = f1
                        all_rows.append({
                            'pathogen':   pathogen,
                            'approach':   label,
                            'k':          k,
                            'classifier': clf_name,
                            'f1_mean':    round(f1, 4),
                            'f1_std':     round(res['f1_std'], 4),
                            'roc_mean':   round(res['roc_mean'], 4) if res['roc_mean'] else None,
                            'n_pos':      n_pos,
                        })

                f_str = lambda r: f"{r['f1_mean']:.4f}" if r else ' N/A '
                print(f'  K={k} {clf_name:<22} A={f_str(res_a)} B={f_str(res_b)} C={f_str(res_c)} D={f_str(res_d)}')

        # ── ABORDAGEM E: STACKING ─────────────────────────────────────────────
        print(f'  Stacking (K=2..5 base + LR meta):')
        for clf_name in ['GaussianNB', 'SVC_RBF', 'LogisticRegression']:
            res_e = cv_stacking(all_seqs, y, clf_name)
            if res_e:
                f1 = res_e['f1_mean']
                if f1 > best_per_approach['E_stacking']:
                    best_per_approach['E_stacking'] = f1
                all_rows.append({
                    'pathogen': pathogen, 'approach': 'E_stacking', 'k': 0,
                    'classifier': f'stack_{clf_name}', 'f1_mean': round(f1, 4),
                    'f1_std': round(res_e['f1_std'], 4),
                    'roc_mean': round(res_e['roc_mean'], 4) if res_e['roc_mean'] else None,
                    'n_pos': n_pos,
                })
                print(f'    {clf_name}: F1={f1:.4f}')

        # ── Resumo do patógeno ────────────────────────────────────────────────
        print(f'\n  === Resumo {pathogen} ===')
        print(f'  v1 referência: {v1_ref:.4f}')
        for ap, f1 in best_per_approach.items():
            delta = f1 - v1_ref
            marker = '★ MELHOR' if f1 == max(best_per_approach.values()) else ''
            print(f'  {ap:<15} F1={f1:.4f}  Δ={delta:+.4f}  {marker}')

        # Salva parcial
        pd.DataFrame(all_rows).to_csv(str(OUT_DIR / 'resultados_parciais.csv'), index=False)

    # ── Relatório final ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(str(OUT_DIR / 'resultados_completos.csv'), index=False)

    print('\n' + '='*80)
    print('  RESULTADO FINAL — Melhor F1 por abordagem (média sobre patógenos)')
    print('='*80)

    v1_vals = list(v1_best.values())
    approaches = ['A_baseline', 'B_positional', 'C_padj', 'D_pos_padj', 'E_stacking']
    summary_rows = []
    for ap in approaches:
        sub = df[df['approach'] == ap]
        best_per_path = sub.groupby('pathogen')['f1_mean'].max()
        mean_f1 = best_per_path.mean()
        n_above_70 = (best_per_path >= 0.70).sum()
        n_beats_v1 = sum(best_per_path.get(p, 0) > v1_best.get(p, 0) for p in PATHOGENS)
        summary_rows.append({'Abordagem': ap, 'F1 médio': round(mean_f1, 4),
                             'F1≥0.70': int(n_above_70), 'Melhora vs v1': int(n_beats_v1)})
        print(f'  {ap:<15}  F1 médio={mean_f1:.4f}  F1≥0.70: {n_above_70}/12  bate v1: {n_beats_v1}/12')

    pd.DataFrame(summary_rows).to_csv(str(OUT_DIR / 'resumo_abordagens.csv'), index=False)

    elapsed = time.time() - t0
    from datetime import timedelta
    print(f'\n  Tempo total: {timedelta(seconds=int(elapsed))}')
    print(f'  Resultados em: {OUT_DIR}/')

    # Melhor abordagem por patógeno
    print('\n  VENCEDOR POR PATÓGENO:')
    for pathogen in PATHOGENS:
        sub = df[df['pathogen'] == pathogen]
        if sub.empty: continue
        best = sub.loc[sub['f1_mean'].idxmax()]
        v1   = v1_best.get(pathogen, 0)
        print(f'  {pathogen:<35} {best["approach"]:<15} K={best["k"]} '
              f'{best["classifier"]:<20} F1={best["f1_mean"]:.4f}  Δv1={best["f1_mean"]-v1:+.4f}')

if __name__ == '__main__':
    main()
