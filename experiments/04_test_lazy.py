#!/usr/bin/env python3
"""
test_lazy.py — Duas etapas de triagem:

  Etapa 1 (LazyPredict, K=4 fixo):
    Varre ~30 classificadores sklearn para identificar candidatos novos.
    Resultados: experimento_lazy/resultados_lazy.csv
                experimento_lazy/top5_por_patogeno.csv

  Etapa 2 (QDA grid K=2..6):
    Testa QuadraticDiscriminantAnalysis — melhor candidato novo —
    com K=2,3,4,5,6 para encontrar o K ótimo por patógeno.
    Resultados: experimento_lazy/qda_grid_k.csv

NÃO modifica nenhum modelo salvo.
"""

import glob, random, warnings, sys
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lazypredict.Supervised import LazyClassifier

warnings.filterwarnings('ignore')

BASE_PATH  = Path('/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data')
FASTA_PATH = BASE_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SIG_BASE   = BASE_PATH / 'Biotic stress/significant_results'
OUT_DIR    = Path('experimento_lazy')
OUT_DIR.mkdir(exist_ok=True)

K_FIXED    = 4        # k-mer fixo para triagem LazyPredict
K_VALUES   = [2, 3, 4, 5, 6]
N_SEL      = 100
N_PCA      = 50
NEG_RATIO  = 1
N_SPLITS   = 5

PATHOGENS = [
    'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
    'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
    'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
    'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
]

KNOWN_CLFS = {
    'GaussianNB', 'BernoulliNB', 'SVC', 'NuSVC',
    'LogisticRegression', 'KNeighborsClassifier',
    'RandomForestClassifier', 'MLPClassifier',
}

def log(msg, **kw):
    print(msg, flush=True, **kw)


# ── dados ─────────────────────────────────────────────────────────────────────

def load_all_lncrnas():
    return {r.id: str(r.seq).upper().replace('U', 'T')
            for r in SeqIO.parse(str(FASTA_PATH), 'fasta')}

def load_pathogen_ids(pathogen):
    csvs = glob.glob(str(SIG_BASE / pathogen / 'lncRNA/Differential expression analysis/*.csv'))
    if not csvs:
        return set(), set()
    df = pd.read_csv(csvs[0])
    all_de = set(df['lncRNA_transcript_ID'].astype(str))
    pos    = set(df.loc[df['log2FoldChange'] > 1, 'lncRNA_transcript_ID'].astype(str))
    return pos, all_de

def build_dataset(pathogen, all_lncrnas, seed=42):
    pos_ids, all_de_ids = load_pathogen_ids(pathogen)
    pos_valid = [p for p in pos_ids if p in all_lncrnas]
    neg_pool  = [s for s in all_lncrnas if s not in all_de_ids]
    random.seed(seed)
    neg_ids = random.sample(neg_pool, min(len(neg_pool), len(pos_valid) * NEG_RATIO))
    return [all_lncrnas[p] for p in pos_valid], [all_lncrnas[n] for n in neg_ids]


# ── features ──────────────────────────────────────────────────────────────────

def count_kmers(seq, k):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    total = len(kmers)
    if total == 0:
        return {}
    c = Counter(kmers)
    return {km: cnt/total for km, cnt in c.items()}

def build_feature_matrix(sequences, k, kmer_dict=None):
    if kmer_dict is None:
        vocab = set()
        for seq in sequences:
            for i in range(len(seq)-k+1):
                vocab.add(seq[i:i+k])
        kmer_dict = {km: i for i, km in enumerate(sorted(vocab))}
    sorted_km = sorted(kmer_dict)
    rows = [[count_kmers(seq, k).get(km, 0.0) for km in sorted_km] for seq in sequences]
    return np.array(rows, dtype=np.float32), kmer_dict

def fit_preprocess(X_tr, y_tr, X_te):
    k_sel    = min(N_SEL, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k_sel).fit(X_tr, y_tr)
    X_tr_s   = selector.transform(X_tr)
    X_te_s   = selector.transform(X_te)
    n_pca    = min(N_PCA, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
    pca      = PCA(n_components=n_pca).fit(X_tr_s)
    X_tr_p   = pca.transform(X_tr_s)
    X_te_p   = pca.transform(X_te_s)
    scaler   = StandardScaler().fit(X_tr_p)
    return scaler.transform(X_tr_p), scaler.transform(X_te_p)


# ── avaliação lazy com CV ─────────────────────────────────────────────────────

def qda_cv(X, y):
    """5-fold CV com QuadraticDiscriminantAnalysis. Retorna (f1_mean, f1_std)."""
    n_pos  = int(np.sum(y == 1))
    actual = min(N_SPLITS, n_pos)
    if actual < 2:
        return None
    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        try:
            X_tr_f, X_te_f = fit_preprocess(X[tr_idx], y[tr_idx], X[te_idx])
            clf = QuadraticDiscriminantAnalysis()
            clf.fit(X_tr_f, y[tr_idx])
            scores.append(f1_score(y[te_idx], clf.predict(X_te_f), zero_division=0))
        except Exception:
            continue
    if not scores:
        return None
    return float(np.mean(scores)), float(np.std(scores))


def lazy_cv(X, y, pathogen):
    """
    Para cada fold: pré-processa dentro do fold, roda LazyClassifier,
    coleta F1 por modelo. Retorna DataFrame com média e std de F1 por modelo.
    """
    n_pos  = int(np.sum(y == 1))
    actual = min(N_SPLITS, n_pos)
    if actual < 2:
        return None

    skf = StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)
    fold_f1s: dict[str, list[float]] = {}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr_f, X_te_f = fit_preprocess(X[tr_idx], y[tr_idx], X[te_idx])
        y_tr, y_te = y[tr_idx], y[te_idx]

        try:
            clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)
            models, preds_df = clf.fit(X_tr_f, X_te_f, y_tr, y_te)
        except Exception as e:
            log(f'    fold {fold} erro: {e}')
            continue

        # preds_df: DataFrame com colunas = nome do modelo, linhas = amostras de teste
        for model_name in preds_df.columns:
            try:
                preds = preds_df[model_name].values
                f1 = f1_score(y_te, preds, zero_division=0)
                fold_f1s.setdefault(model_name, []).append(f1)
            except Exception:
                continue

    if not fold_f1s:
        return None

    rows = []
    for model_name, scores in fold_f1s.items():
        if len(scores) < 2:
            continue
        rows.append({
            'pathogen':   pathogen,
            'model':      model_name,
            'f1_mean':    round(float(np.mean(scores)), 4),
            'f1_std':     round(float(np.std(scores)), 4),
            'n_folds':    len(scores),
            'is_new':     model_name not in KNOWN_CLFS,
        })

    return pd.DataFrame(rows).sort_values('f1_mean', ascending=False)


# ── main ──────────────────────────────────────────────────────────────────────

def run_lazy(all_lncrnas):
    """Etapa 1: LazyPredict com K fixo=4."""
    log('=' * 70)
    log(f'  ETAPA 1 — LazyPredict  K={K_FIXED}, 5-fold CV')
    log('=' * 70)

    all_results, top5_rows = [], []

    for p_idx, pathogen in enumerate(PATHOGENS, 1):
        log(f'\n[{p_idx:02d}/{len(PATHOGENS)}] {pathogen}')
        pos_seqs, neg_seqs = build_dataset(pathogen, all_lncrnas)
        n_pos, n_neg = len(pos_seqs), len(neg_seqs)
        log(f'  Dataset: {n_pos} pos | {n_neg} neg')

        if n_pos < 4:
            log('  ⚠ Insuficiente, pulando.')
            continue

        all_seqs = pos_seqs + neg_seqs
        y        = np.array([1]*n_pos + [0]*n_neg)
        X, _     = build_feature_matrix(all_seqs, K_FIXED)

        df_lazy = lazy_cv(X, y, pathogen)
        if df_lazy is None or df_lazy.empty:
            log('  ⚠ Sem resultados.')
            continue

        all_results.append(df_lazy)
        top5 = df_lazy.head(5).copy()
        log('  Top-5:')
        for _, row in top5.iterrows():
            tag = ' ★NEW' if row['is_new'] else ''
            log(f'    {row["model"]:<40} F1={row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}{tag}')
        top5['rank'] = range(1, len(top5)+1)
        top5_rows.append(top5)

    if not all_results:
        log('Nenhum resultado na Etapa 1.')
        return

    df_all  = pd.concat(all_results,  ignore_index=True)
    df_top5 = pd.concat(top5_rows,    ignore_index=True)
    df_all.to_csv(OUT_DIR / 'resultados_lazy.csv',    index=False)
    df_top5.to_csv(OUT_DIR / 'top5_por_patogeno.csv', index=False)

    log('\n' + '=' * 70)
    log('  CANDIDATOS NOVOS no top-3')
    log('=' * 70)
    top3     = df_top5[df_top5['rank'] <= 3]
    new_top3 = top3[top3['is_new']].groupby('model').agg(
        pathogens_top3=('pathogen', 'count'),
        f1_mean_avg=('f1_mean', 'mean'),
    ).sort_values('pathogens_top3', ascending=False)

    if new_top3.empty:
        log('  Nenhum modelo novo no top-3.')
    else:
        log(new_top3.to_string())
        log('\n  → Candidatos para retrain_v4.py:')
        for model, row in new_top3.iterrows():
            if row['pathogens_top3'] >= 3:
                log(f'    {model}  (top-3: {int(row["pathogens_top3"])}/12  F1 médio={row["f1_mean_avg"]:.4f})')


def run_qda_grid(all_lncrnas):
    """Etapa 2: QDA com K=2..6 — busca o K ótimo por patógeno."""
    log('\n' + '=' * 70)
    log(f'  ETAPA 2 — QDA grid K={K_VALUES}, 5-fold CV')
    log('=' * 70)

    # carrega referência v4 atual
    v4_ref = {}
    v4_csv = Path('grid_search_results_v4/melhores_modelos.csv')
    if v4_csv.exists():
        df_ref = pd.read_csv(v4_csv)
        v4_ref = dict(zip(df_ref['pathogen'], df_ref['f1']))

    rows = []
    total = len(PATHOGENS) * len(K_VALUES)
    done  = 0

    for pathogen in PATHOGENS:
        log(f'\n  {pathogen}')
        pos_seqs, neg_seqs = build_dataset(pathogen, all_lncrnas)
        if len(pos_seqs) < 4:
            log('    ⚠ Insuficiente.')
            continue

        all_seqs = pos_seqs + neg_seqs
        y        = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
        best_k, best_f1 = None, -1

        for k in K_VALUES:
            done += 1
            X, _ = build_feature_matrix(all_seqs, k)
            res  = qda_cv(X, y)
            if res is None:
                continue
            f1_mean, f1_std = res
            v4_f1  = v4_ref.get(pathogen, 0)
            delta  = f1_mean - v4_f1
            flag   = ' ▲ MELHORA' if delta > 0.005 else (' ▼' if delta < -0.005 else '')
            log(f'    K={k}  QDA  F1={f1_mean:.4f} ± {f1_std:.4f}  Δv4={delta:+.4f}{flag}  [{done}/{total}]')
            rows.append({
                'pathogen': pathogen, 'k': k,
                'f1_qda': round(f1_mean, 4), 'f1_std': round(f1_std, 4),
                'f1_v4': round(v4_f1, 4), 'delta_v4': round(delta, 4),
            })
            if f1_mean > best_f1:
                best_f1, best_k = f1_mean, k

        if best_k is not None:
            v4_f1 = v4_ref.get(pathogen, 0)
            log(f'    ★ Melhor QDA: K={best_k}  F1={best_f1:.4f}  (v4={v4_f1:.4f}  Δ={best_f1-v4_f1:+.4f})')

    df_qda = pd.DataFrame(rows)
    df_qda.to_csv(OUT_DIR / 'qda_grid_k.csv', index=False)

    log('\n' + '=' * 70)
    log('  RESUMO QDA — melhor K por patógeno vs v4 atual')
    log('=' * 70)
    if not df_qda.empty:
        best_per = df_qda.loc[df_qda.groupby('pathogen')['f1_qda'].idxmax()]
        best_per = best_per.sort_values('delta_v4', ascending=False)
        log(best_per[['pathogen','k','f1_qda','f1_v4','delta_v4']].to_string(index=False))
        ganhou = (best_per['delta_v4'] > 0.005).sum()
        log(f'\n  QDA melhora v4 em {ganhou}/{len(best_per)} patógenos')
        log(f'  → Resultados em {OUT_DIR}/qda_grid_k.csv')


def main():
    log('Carregando sequências...')
    all_lncrnas = load_all_lncrnas()
    log(f'  ✓ {len(all_lncrnas)} lncRNAs')

    run_lazy(all_lncrnas)
    run_qda_grid(all_lncrnas)


if __name__ == '__main__':
    main()
