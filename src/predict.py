#!/usr/bin/env python3
"""
Aplicação web para predição de patógenos a partir de sequência lncRNA
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import joblib
import os
import random
from pathlib import Path
from collections import Counter
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── k-mers canônicos (v2) ─────────────────────────────────────────────────────
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
_VALID_NTS  = frozenset('ATCG')

def _revcomp(seq):
    return ''.join(_COMPLEMENT.get(b, 'N') for b in reversed(seq))

def _canonical(kmer):
    rc = _revcomp(kmer)
    return kmer if kmer <= rc else rc

# Valores padrão (antes do primeiro retreino). Sobrescritos pelo JSON se existir.
_DEFAULT_MODELS_INFO = {
    'Alternaria solani':          {'model_type': 'MLP_Shallow',          'k': 6, 'f1': 0.8824},
    'Meloidogyne javanica':       {'model_type': 'ExtraTreesClassifier', 'k': 6, 'f1': 0.9474},
    'Pectobacterium carotovorum': {'model_type': 'NuSVC',                'k': 6, 'f1': 0.8378},
    'Spongospora subterranea':    {'model_type': 'Ridge',                'k': 6, 'f1': 0.8718},
    'Globodera spp':              {'model_type': 'KNeighborsClassifier', 'k': 6, 'f1': 0.7218},
    'Leptinotarsa decemlineata':  {'model_type': 'SVC_RBF',             'k': 5, 'f1': 0.7174},
    'Phytophthora infestans':     {'model_type': 'BernoulliNB',         'k': 3, 'f1': 0.7161},
    'Ralstonia solanacearum':     {'model_type': 'ExtraTreesClassifier', 'k': 6, 'f1': 0.6977},
    'Streptomyces scabies':       {'model_type': 'LogisticRegression',  'k': 6, 'f1': 0.7104},
    'Synchytrium endobioticum':   {'model_type': 'BaggingClassifier',   'k': 3, 'f1': 0.7769},
    'Potato virus A':             {'model_type': 'BernoulliNB',         'k': 2, 'f1': 0.6624},
    'Potato virus Y':             {'model_type': 'SVC_RBF',             'k': 4, 'f1': 0.6981},
}

_CONFIG_JSON = Path(__file__).parent / 'best_models_config.json'

# Flags globais — lidas do config JSON gerado pelo script de treino
USE_CANONICAL_KMERS = False
USE_PCA             = True
USE_BIO_FEATURES    = False   # v3 concatena features biológicas antes dos k-mers
N_BIO               = 10

def _load_models_info():
    global USE_CANONICAL_KMERS, USE_PCA, USE_BIO_FEATURES, N_BIO
    if _CONFIG_JSON.exists():
        try:
            with open(_CONFIG_JSON, encoding='utf-8') as f:
                cfg = json.load(f)
            USE_CANONICAL_KMERS = cfg.get('_canonical_kmers', False)
            USE_PCA             = cfg.get('_pca', True)
            USE_BIO_FEATURES    = cfg.get('_bio_features', False)
            N_BIO               = cfg.get('_n_bio', 10)
            version             = cfg.get('_version', 'v1')
            merged = {}
            for p, defaults in _DEFAULT_MODELS_INFO.items():
                if p in cfg and isinstance(cfg[p], dict):
                    merged[p] = {**defaults, **cfg[p]}
                else:
                    merged[p] = defaults
            print(f'  Config: versão={version} | canonical={USE_CANONICAL_KMERS} | PCA={USE_PCA} | bio_features={USE_BIO_FEATURES}')
            return merged
        except Exception as e:
            print(f'  ⚠ Erro ao ler {_CONFIG_JSON}: {e} — usando defaults')
    return dict(_DEFAULT_MODELS_INFO)

BEST_MODELS_INFO = _load_models_info()

PATHOGEN_INFO = {
    'Alternaria solani':          {'tipo': 'Fungo',           'doenca': 'Pinta preta',            'icone': '🍄'},
    'Globodera spp':              {'tipo': 'Nematoide',       'doenca': 'Nematoide de cisto',     'icone': '🪱'},
    'Leptinotarsa decemlineata':  {'tipo': 'Inseto',          'doenca': 'Besouro da batata',      'icone': '🐛'},
    'Meloidogyne javanica':       {'tipo': 'Nematoide',       'doenca': 'Nematoide das galhas',   'icone': '🪱'},
    'Pectobacterium carotovorum': {'tipo': 'Bactéria',        'doenca': 'Podridão mole',          'icone': '🦠'},
    'Phytophthora infestans':     {'tipo': 'Oomiceto',        'doenca': 'Requeima',               'icone': '🍂'},
    'Potato virus A':             {'tipo': 'Vírus',           'doenca': 'Vírus A da batata',      'icone': '🧬'},
    'Potato virus Y':             {'tipo': 'Vírus',           'doenca': 'Vírus Y da batata',      'icone': '🧬'},
    'Ralstonia solanacearum':     {'tipo': 'Bactéria',        'doenca': 'Murcha bacteriana',      'icone': '🦠'},
    'Spongospora subterranea':    {'tipo': 'Protista',        'doenca': 'Sarna pulverulenta',     'icone': '🌿'},
    'Streptomyces scabies':       {'tipo': 'Actinobactéria',  'doenca': 'Sarna comum',            'icone': '🦠'},
    'Synchytrium endobioticum':   {'tipo': 'Fungo',           'doenca': 'Verruga da batata',      'icone': '🍄'},
}

MODELS_CACHE = {}
PREPROCESSORS_CACHE = {}
LNCRNA_SEQUENCES = {}  # id → sequence string

LNCRNA_FASTA = (
    Path(__file__).parent /
    'potato_data/Browse/Browse_sequence/lncRNA.fa'
)


def load_lncrna_fasta():
    if not LNCRNA_FASTA.exists():
        print(f"  ⚠ FASTA não encontrado em {LNCRNA_FASTA}")
        return
    for record in SeqIO.parse(str(LNCRNA_FASTA), 'fasta'):
        LNCRNA_SEQUENCES[record.id] = str(record.seq)
    print(f"  ✓ {len(LNCRNA_SEQUENCES)} sequências lncRNA carregadas")


def load_models():
    print("Carregando modelos...")
    for pathogen in BEST_MODELS_INFO:
        name = pathogen.replace(' ', '_')
        model_path = Path(f'modelos_salvos/{name}_model.pkl')
        if not model_path.exists():
            print(f"  ⚠ {pathogen}: arquivo não encontrado")
            continue
        try:
            pca_path = Path(f'modelos_salvos/preprocessadores/{name}_pca.pkl')
            pca_obj  = joblib.load(str(pca_path)) if pca_path.exists() else None

            MODELS_CACHE[pathogen] = joblib.load(model_path)
            PREPROCESSORS_CACHE[pathogen] = {
                'selector':  joblib.load(f'modelos_salvos/preprocessadores/{name}_selector.pkl'),
                'pca':       pca_obj,   # None se v2 (sem PCA)
                'scaler':    joblib.load(f'modelos_salvos/preprocessadores/{name}_scaler.pkl'),
                'kmer_dict': joblib.load(f'modelos_salvos/preprocessadores/{name}_kmer_dict.pkl'),
            }
            pca_note = 'sem PCA' if pca_obj is None else 'com PCA'
            print(f"  ✓ {pathogen} ({pca_note})")
        except Exception as e:
            print(f"  ✗ {pathogen}: {e}")


# ── feature extraction ───────────────────────────────────────────────────────

def bio_features(seq: str) -> list:
    """10 features biológicas (mesma ordem do retrain_v3.py)."""
    n = len(seq)
    if n == 0:
        return [0.0] * 10
    c = Counter(seq)
    A   = c.get('A', 0) / n
    T   = c.get('T', 0) / n
    G   = c.get('G', 0) / n
    C_f = c.get('C', 0) / n
    gc  = G + C_f
    at  = A + T
    cg_obs = sum(1 for i in range(n - 1) if seq[i:i+2] == 'CG') / max(n - 1, 1)
    cg_exp = C_f * G
    cpg    = cg_obs / max(cg_exp, 1e-10)
    entropy = -sum(p * np.log2(p + 1e-10) for p in [A, T, G, C_f] if p > 0)
    length_log = np.log1p(n)
    gc_denom   = G + C_f
    skew_gc    = (G - C_f) / max(gc_denom, 1e-10)
    at_denom   = A + T
    skew_at    = (A - T)  / max(at_denom, 1e-10)
    purine     = A + G
    return [gc, at, cpg, entropy, length_log, skew_gc, skew_at, purine, A, T]


def count_kmers(seq, k):
    """Frequência relativa de k-mers simples (forward strand — compatibilidade v1)."""
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    total = len(kmers)
    if total == 0:
        return Counter()
    counts = Counter(kmers)
    return Counter({km: c / total for km, c in counts.items()})


def count_kmers_canonical(seq, k):
    """Frequência relativa de k-mers canônicos: min(kmer, revcomp(kmer)) — v2."""
    counts: Counter = Counter()
    total = 0
    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if _VALID_NTS.issuperset(km):
            counts[_canonical(km)] += 1
            total += 1
    if total == 0:
        return Counter()
    return Counter({km: c / total for km, c in counts.items()})


def sequence_to_vector(sequence, k, kmer_dict):
    seq = sequence.upper().replace('U', 'T')
    counts = count_kmers_canonical(seq, k) if USE_CANONICAL_KMERS else count_kmers(seq, k)
    kmer_row = [counts.get(km, 0) for km in sorted(kmer_dict)]
    if USE_BIO_FEATURES:
        row = bio_features(seq) + kmer_row
    else:
        row = kmer_row
    return np.array(row, dtype=np.float32).reshape(1, -1)


def validate_sequence(raw):
    seq = raw.strip()
    # aceitar FASTA (remove cabeçalho)
    if seq.startswith('>'):
        lines = seq.splitlines()
        seq = ''.join(l.strip() for l in lines if not l.startswith('>'))
    seq = seq.upper().replace(' ', '').replace('\n', '').replace('\r', '')
    if not seq:
        return False, "Sequência vazia."
    invalid = set(seq) - set('ATCGURN')
    if invalid:
        return False, f"Caracteres inválidos: {', '.join(sorted(invalid))}. Use A T C G U N."
    if len(seq) < 20:
        return False, "Sequência muito curta (mínimo 20 nt)."
    return True, seq


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_probability(model, X_scaled):
    """Retorna probabilidade da classe positiva (0–1)."""
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X_scaled)[0][1])
    # Ridge regressor: resultado contínuo, mapeamos com sigmoid
    raw = float(model.predict(X_scaled)[0])
    return float(np.clip(sigmoid(raw), 0.0, 1.0))


# ── rotas ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/info')
def info():
    pathogens_out = {}
    for p, m in BEST_MODELS_INFO.items():
        pathogens_out[p] = {
            **m,
            **PATHOGEN_INFO.get(p, {}),
            'loaded': p in MODELS_CACHE,
        }
    return jsonify({
        'total_pathogens': len(BEST_MODELS_INFO),
        'models_loaded': len(MODELS_CACHE),
        'pathogens': pathogens_out,
    })


@app.route('/api/random')
def random_sequence():
    """Retorna uma sequência lncRNA aleatória do banco de dados."""
    if not LNCRNA_SEQUENCES:
        return jsonify({'error': 'Banco de sequências não carregado.'}), 500
    seq_id = random.choice(list(LNCRNA_SEQUENCES.keys()))
    seq = LNCRNA_SEQUENCES[seq_id]
    return jsonify({'id': seq_id, 'sequence': seq, 'length': len(seq)})


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        raw = data.get('sequence', '')

        valid, result = validate_sequence(raw)
        if not valid:
            return jsonify({'error': result}), 400
        sequence = result

        if not MODELS_CACHE:
            return jsonify({'error': 'Nenhum modelo carregado. Execute train_and_save_models_v2.py'}), 500

        predictions = {}
        for pathogen, m_info in BEST_MODELS_INFO.items():
            if pathogen not in MODELS_CACHE:
                predictions[pathogen] = {
                    'probabilidade': None,
                    'f1_modelo': m_info['f1'],
                    'status': 'modelo_indisponivel',
                }
                continue
            try:
                prep = PREPROCESSORS_CACHE[pathogen]
                k = m_info['k']
                X = sequence_to_vector(sequence, k, prep['kmer_dict'])
                X = prep['selector'].transform(X)
                if prep['pca'] is not None:   # v1 usava PCA; v2 não usa
                    X = prep['pca'].transform(X)
                X = prep['scaler'].transform(X)
                prob = get_probability(MODELS_CACHE[pathogen], X)
                predictions[pathogen] = {
                    'probabilidade': round(prob * 100, 2),
                    'f1_modelo': m_info['f1'],
                    'status': 'ok',
                }
            except Exception as e:
                predictions[pathogen] = {
                    'probabilidade': None,
                    'f1_modelo': m_info['f1'],
                    'status': f'erro: {e}',
                }

        valid_preds = {p: v for p, v in predictions.items() if v['status'] == 'ok'}
        if not valid_preds:
            return jsonify({'error': 'Nenhuma predição disponível.', 'predictions': predictions}), 200

        top = max(valid_preds, key=lambda p: valid_preds[p]['probabilidade'])
        return jsonify({
            'sequence_length': len(sequence),
            'predictions': predictions,
            'top_prediction': {
                'pathogen': top,
                'probabilidade': valid_preds[top]['probabilidade'],
                'f1_modelo': valid_preds[top]['f1_modelo'],
                **PATHOGEN_INFO.get(top, {}),
            },
            'pathogen_info': PATHOGEN_INFO,
        })

    except Exception as e:
        return jsonify({'error': f'Erro interno: {e}'}), 500


if __name__ == '__main__':
    print('=' * 70)
    print('BiSPoLP — Biotic Stress Potato LncRNA Predictor')
    print('=' * 70)
    load_lncrna_fasta()
    load_models()
    print(f'\n✅ {len(MODELS_CACHE)}/{len(BEST_MODELS_INFO)} modelos carregados')
    print('Acesse: http://localhost:5000\n')
    app.run(debug=True, port=5000)
