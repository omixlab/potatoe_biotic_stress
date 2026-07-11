#!/usr/bin/env python3
"""
BATATA — Biotic Attack Targeting Algorithm for Tuberosum Analysis
Aplicação web Flask para predição de lncRNAs responsivos a patógenos de batata.

Como rodar:
    python app.py
Acesse: http://localhost:5000
"""

import json, os, random, time, warnings
from pathlib import Path
from collections import Counter, defaultdict
from functools import wraps

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from Bio import SeqIO

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Segurança ──────────────────────────────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024   # 512 KB máximo por requisição

# Rate limiting simples em memória: max 10 predições por IP por minuto
_rate_store: dict = defaultdict(list)
RATE_LIMIT   = 10
RATE_WINDOW  = 60  # segundos

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip  = request.remote_addr or 'unknown'
        now = time.time()
        hits = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return jsonify({'error': 'Muitas requisições. Tente novamente em breve.'}), 429
        hits.append(now)
        _rate_store[ip] = hits
        return f(*args, **kwargs)
    return decorated

@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options']  = 'nosniff'
    response.headers['X-Frame-Options']         = 'DENY'
    response.headers['X-XSS-Protection']        = '1; mode=block'
    response.headers['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / 'models' / 'saved'
CONFIG_JSON = ROOT / 'models' / 'best_models_config.json'

# FASTA de lncRNAs — ajuste DATA_PATH se os dados estiverem em outro local.
# Se o dataset completo não estiver disponível, usa o FASTA de amostra
# embutido no repositório (data/sample_lncRNAs.fa) para o botão "aleatório".
DATA_PATH = Path(os.environ.get(
    'BATATA_DATA_PATH',
    '/home/christian/Documentos/projeto_biotic_stress_potatoe/potato_data'
))
LNCRNA_FASTA = DATA_PATH / 'Browse/Browse_sequence/lncRNA.fa'
SAMPLE_FASTA = ROOT / 'data' / 'sample_lncRNAs.fa'

# ── Informações estáticas por patógeno ────────────────────────────────────────
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

# ── Carrega configuração dos modelos ──────────────────────────────────────────
_DEFAULT_INFO = {p: {'k': 4, 'model_type': 'GaussianNB', 'f1': 0.0} for p in PATHOGEN_INFO}
USE_PCA = True

def _load_config():
    global USE_PCA
    if CONFIG_JSON.exists():
        try:
            cfg = json.loads(CONFIG_JSON.read_text(encoding='utf-8'))
            USE_PCA = cfg.get('_pca', True)
            info = {}
            for p in PATHOGEN_INFO:
                if p in cfg and isinstance(cfg[p], dict):
                    info[p] = {**_DEFAULT_INFO[p], **cfg[p]}
                else:
                    info[p] = _DEFAULT_INFO[p]
            print(f'  Configuração carregada: versão={cfg.get("_version","?")} | PCA={USE_PCA}')
            return info
        except Exception as e:
            print(f'  Aviso: erro ao ler config ({e}) — usando defaults')
    return dict(_DEFAULT_INFO)

MODELS_INFO = _load_config()

# ── Cache de modelos e pré-processadores ──────────────────────────────────────
MODELS      = {}
PREPROCESSORS = {}
SEQUENCES   = {}

def load_models():
    print('Carregando modelos...')
    for pathogen in MODELS_INFO:
        name  = pathogen.replace(' ', '_')
        mpath = MODELS_DIR / f'{name}_model.pkl'
        if not mpath.exists():
            print(f'  ⚠ {pathogen}: modelo não encontrado em {mpath}')
            continue
        try:
            pca_path = MODELS_DIR / f'{name}_pca.pkl'
            MODELS[pathogen] = joblib.load(mpath)
            PREPROCESSORS[pathogen] = {
                'selector':  joblib.load(MODELS_DIR / f'{name}_selector.pkl'),
                'pca':       joblib.load(pca_path) if pca_path.exists() else None,
                'scaler':    joblib.load(MODELS_DIR / f'{name}_scaler.pkl'),
                'kmer_dict': joblib.load(MODELS_DIR / f'{name}_kmer_dict.pkl'),
            }
            print(f'  ✓ {pathogen}')
        except Exception as e:
            print(f'  ✗ {pathogen}: {e}')

def load_sequences():
    # Prefere o dataset completo (BATATA_DATA_PATH); se ausente, usa a amostra
    # embutida no repositório para que o botão "aleatório" funcione out-of-the-box.
    if LNCRNA_FASTA.exists():
        fasta, origem = LNCRNA_FASTA, 'dataset completo'
    elif SAMPLE_FASTA.exists():
        fasta, origem = SAMPLE_FASTA, 'amostra embutida'
    else:
        print(f'  Aviso: nenhum FASTA encontrado ({LNCRNA_FASTA} nem {SAMPLE_FASTA})')
        print(f'  (defina BATATA_DATA_PATH=<pasta> para usar o dataset completo)')
        return
    for rec in SeqIO.parse(str(fasta), 'fasta'):
        SEQUENCES[rec.id] = str(rec.seq)
    print(f'  ✓ {len(SEQUENCES)} sequências carregadas ({origem})')

# ── Extração de k-mers ────────────────────────────────────────────────────────
def kmers(seq: str, k: int) -> Counter:
    total = len(seq) - k + 1
    if total <= 0:
        return Counter()
    c = Counter(seq[i:i+k] for i in range(total))
    return Counter({km: v/total for km, v in c.items()})

def seq_to_vector(sequence: str, k: int, kmer_dict: dict) -> np.ndarray:
    seq = sequence.upper().replace('U', 'T')
    counts = kmers(seq, k)
    return np.array([counts.get(km, 0) for km in sorted(kmer_dict)], dtype=np.float32).reshape(1, -1)

MAX_SEQ_LEN = 50_000  # lncRNAs típicos: 200–10.000 nt

def validate(raw: str):
    if not isinstance(raw, str) or len(raw) > MAX_SEQ_LEN * 2:
        return False, 'Sequência muito longa ou formato inválido.'
    seq = raw.strip()
    if seq.startswith('>'):
        lines = seq.splitlines()
        seq = ''.join(l for l in lines if not l.startswith('>'))
    seq = seq.upper().replace(' ', '').replace('\n', '').replace('\r', '')
    if not seq:
        return False, 'Sequência vazia.'
    if len(seq) < 20:
        return False, 'Sequência muito curta (mínimo 20 nt).'
    if len(seq) > MAX_SEQ_LEN:
        return False, f'Sequência muito longa (máximo {MAX_SEQ_LEN} nt).'
    invalid = set(seq) - set('ATCGURN')
    if invalid:
        return False, f"Caracteres inválidos: {', '.join(sorted(invalid))}"
    return True, seq

def get_prob(model, X) -> float:
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0][1])
    raw = float(model.predict(X)[0])
    return float(np.clip(1 / (1 + np.exp(-raw)), 0, 1))

# ── Rotas ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/info')
def info():
    out = {}
    for p, m in MODELS_INFO.items():
        out[p] = {**m, **PATHOGEN_INFO.get(p, {}), 'loaded': p in MODELS}
    return jsonify({'total_pathogens': len(MODELS_INFO), 'models_loaded': len(MODELS), 'pathogens': out})

@app.route('/api/random')
def random_seq():
    if not SEQUENCES:
        return jsonify({'error': 'Banco de sequências não carregado.'}), 500
    sid = random.choice(list(SEQUENCES))
    return jsonify({'id': sid, 'sequence': SEQUENCES[sid], 'length': len(SEQUENCES[sid])})

@app.route('/api/predict', methods=['POST'])
@rate_limit
def predict():
    try:
        body = request.get_json(silent=True) or {}
        raw  = body.get('sequence', '')
        ok, result = validate(raw)
        if not ok:
            return jsonify({'error': result}), 400
        seq = result

        if not MODELS:
            return jsonify({'error': 'Nenhum modelo carregado.'}), 503

        predictions = {}
        for pathogen, info in MODELS_INFO.items():
            if pathogen not in MODELS:
                predictions[pathogen] = {'probabilidade': None, 'f1_modelo': info['f1'], 'status': 'indisponivel'}
                continue
            try:
                prep = PREPROCESSORS[pathogen]
                X    = seq_to_vector(seq, info['k'], prep['kmer_dict'])
                X    = prep['selector'].transform(X)
                if prep['pca'] is not None:
                    X = prep['pca'].transform(X)
                X    = prep['scaler'].transform(X)
                predictions[pathogen] = {
                    'probabilidade': round(get_prob(MODELS[pathogen], X) * 100, 2),
                    'f1_modelo': info['f1'],
                    'status': 'ok',
                }
            except Exception:
                predictions[pathogen] = {'probabilidade': None, 'f1_modelo': info['f1'], 'status': 'erro'}

        valid = {p: v for p, v in predictions.items() if v['status'] == 'ok'}
        if not valid:
            return jsonify({'error': 'Nenhuma predição disponível.', 'predictions': predictions}), 200

        top = max(valid, key=lambda p: valid[p]['probabilidade'])
        return jsonify({
            'sequence_length': len(seq),
            'predictions': predictions,
            'top_prediction': {
                'pathogen': top,
                **valid[top],
                **PATHOGEN_INFO.get(top, {}),
            },
            'pathogen_info': PATHOGEN_INFO,
        })
    except Exception:
        return jsonify({'error': 'Erro interno no servidor.'}), 500

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('BATATA — Biotic Attack Targeting Algorithm for Tuberosum Analysis')
    print('=' * 65)
    load_sequences()
    load_models()
    print(f'\n✅ {len(MODELS)}/{len(MODELS_INFO)} modelos prontos')
    print('Acesse: http://localhost:5000\n')
    debug = os.environ.get('BATATA_DEBUG', '0') == '1'
    app.run(debug=debug, port=5000)
