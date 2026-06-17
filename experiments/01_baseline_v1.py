#!/usr/bin/env python3
"""
Retreina APENAS os modelos finais usando o pipeline v1 (o melhor).
Lê grid_search_results/melhores_modelos.csv, retreina cada (pathogen, k, clf)
e salva em modelos_salvos/ + atualiza best_models_config.json.
Usado após v3 sobrescrever os arquivos com piores modelos.
"""
import glob, json, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import joblib
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

BASE_PATH = Path('/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data')
FASTA_PATH = BASE_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SIG_BASE   = BASE_PATH / 'Biotic stress/significant_results'
MODEL_DIR  = Path('modelos_salvos')
PREP_DIR   = MODEL_DIR / 'preprocessadores'

N_SEL, N_PCA, NEG_RATIO = 100, 50, 3

def make_classifiers():
    return {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTrees':         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'AdaBoost':           AdaBoostClassifier(n_estimators=100, random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost':            XGBClassifier(n_estimators=100, verbosity=0, random_state=42, n_jobs=-1, eval_metric='logloss'),
        'CatBoost':           CatBoostClassifier(iterations=100, verbose=False, random_state=42),
        'SVC_RBF':            SVC(kernel='rbf', probability=True, random_state=42),
        'NuSVC':              NuSVC(kernel='rbf', probability=True, random_state=42),
        'BernoulliNB':        BernoulliNB(),
        'GaussianNB':         GaussianNB(),
        'KNN':                KNeighborsClassifier(n_neighbors=5),
        'MLP':                MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }

def load_all_lncrnas():
    seqs = {}
    for rec in SeqIO.parse(str(FASTA_PATH), 'fasta'):
        seqs[rec.id] = str(rec.seq).upper().replace('U','T')
    return seqs

def load_pathogen_ids(pathogen):
    csvs = glob.glob(str(SIG_BASE / pathogen / 'lncRNA/Differential expression analysis/*.csv'))
    if not csvs: return set(), set()
    df = pd.read_csv(csvs[0])
    all_de = set(df['lncRNA_transcript_ID'].astype(str))
    pos    = set(df.loc[df['log2FoldChange'] > 1, 'lncRNA_transcript_ID'].astype(str))
    return pos, all_de

def build_dataset(pathogen, all_lncrnas):
    pos_ids, all_de_ids = load_pathogen_ids(pathogen)
    pos_seqs  = [all_lncrnas[p] for p in pos_ids  if p in all_lncrnas]
    neg_pool  = [s for s in all_lncrnas if s not in all_de_ids]
    random.seed(42)
    neg_seqs  = [all_lncrnas[s] for s in random.sample(neg_pool, min(len(neg_pool), len(pos_seqs)*NEG_RATIO))]
    return pos_seqs, neg_seqs

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
    sorted_kmers = sorted(kmer_dict)
    rows = []
    for seq in sequences:
        d = count_kmers(seq, k)
        rows.append([d.get(km, 0.0) for km in sorted_kmers])
    return np.array(rows, dtype=np.float32), kmer_dict

def fit_preprocess(X_tr, y_tr):
    k_sel    = min(N_SEL, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k_sel).fit(X_tr, y_tr)
    X_s      = selector.transform(X_tr)
    n_pca    = min(N_PCA, X_s.shape[1], X_s.shape[0]-1)
    pca      = PCA(n_components=n_pca).fit(X_s)
    X_p      = pca.transform(X_s)
    scaler   = StandardScaler().fit(X_p)
    X_f      = scaler.transform(X_p)
    return X_f, selector, pca, scaler

def apply_smote(X, y):
    n_pos = int(np.sum(y==1))
    k_nb  = min(3, n_pos-1)
    if k_nb < 1: return X, y
    try:   return SMOTE(random_state=42, k_neighbors=k_nb).fit_resample(X, y)
    except: return X, y

def main():
    print("Carregando sequências...")
    all_lncrnas = load_all_lncrnas()
    print(f"  {len(all_lncrnas)} sequências")

    best_v1 = pd.read_csv('grid_search_results/melhores_modelos.csv')
    config  = {}

    print("\nRETREINANDO modelos v1 (pipeline original — o melhor)...")
    for _, row in best_v1.iterrows():
        pathogen = row['pathogen']
        k        = int(row['k'])
        clf_name = row['model_type']
        f1       = float(row['f1'])
        print(f"  ► {pathogen}  K={k}  {clf_name}  (F1={f1:.4f})")

        pos_seqs, neg_seqs = build_dataset(pathogen, all_lncrnas)
        all_seqs = pos_seqs + neg_seqs
        y        = np.array([1]*len(pos_seqs) + [0]*len(neg_seqs))
        X, kd    = build_feature_matrix(all_seqs, k)

        X_f, selector, pca, scaler = fit_preprocess(X, y)
        X_f, y_f = apply_smote(X_f, y)

        clf = make_classifiers()[clf_name]
        clf.fit(X_f, y_f)

        name = pathogen.replace(' ', '_')
        joblib.dump(clf,      str(MODEL_DIR / f'{name}_model.pkl'))
        joblib.dump(selector, str(PREP_DIR  / f'{name}_selector.pkl'))
        joblib.dump(pca,      str(PREP_DIR  / f'{name}_pca.pkl'))
        joblib.dump(scaler,   str(PREP_DIR  / f'{name}_scaler.pkl'))
        joblib.dump(kd,       str(PREP_DIR  / f'{name}_kmer_dict.pkl'))
        print(f"    ✓ salvo")

        config[pathogen] = {'k': k, 'model_type': clf_name, 'f1': f1}

    full_config = {
        '_version':         'v1',
        '_canonical_kmers': False,
        '_pca':             True,
        '_bio_features':    False,
        '_smote':           True,
        **config,
    }
    with open('best_models_config.json', 'w') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)
    print("\n✅ best_models_config.json restaurado com pipeline v1")
    print("   Média F1:", round(best_v1['f1'].mean(), 4))

if __name__ == '__main__':
    main()
