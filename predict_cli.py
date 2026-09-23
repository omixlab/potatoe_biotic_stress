#!/usr/bin/env python3
"""
predict_cli.py — BATATA command-line interface.

Score potato lncRNA sequences for predicted responsiveness to each of the 12
biotic-stress pathogens, from the terminal (no web server needed). Uses the same
trained models and preprocessing as the web application (app.py).

Examples:
    # a FASTA file, table written to stdout
    python predict_cli.py sequences.fa

    # save to a CSV file, only show each sequence's top pathogen
    python predict_cli.py sequences.fa -o results.csv --top

    # a single raw sequence
    python predict_cli.py --sequence ATCGATCG...

    # from stdin
    cat sequences.fa | python predict_cli.py -
"""
import argparse
import csv
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import joblib

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models' / 'saved'
CONFIG_JSON = ROOT / 'models' / 'best_models_config.json'
MAX_SEQ_LEN = 50_000

PATHOGENS = [
    'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
    'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
    'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
    'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
]


# ── loading ────────────────────────────────────────────────────────────────────
def load_config():
    import json
    cfg = json.loads(CONFIG_JSON.read_text(encoding='utf-8'))
    info = {}
    for p in PATHOGENS:
        if isinstance(cfg.get(p), dict):
            info[p] = {'k': cfg[p].get('k', 4), 'f1': cfg[p].get('f1', 0.0)}
        else:
            info[p] = {'k': 4, 'f1': 0.0}
    return info


def load_models(info):
    models, preps = {}, {}
    for pathogen in info:
        name = pathogen.replace(' ', '_')
        mpath = MODELS_DIR / f'{name}_model.pkl'
        if not mpath.exists():
            print(f'  ⚠ {pathogen}: model not found ({mpath})', file=sys.stderr)
            continue
        try:
            pca_path = MODELS_DIR / f'{name}_pca.pkl'
            models[pathogen] = joblib.load(mpath)
            preps[pathogen] = {
                'selector':  joblib.load(MODELS_DIR / f'{name}_selector.pkl'),
                'pca':       joblib.load(pca_path) if pca_path.exists() else None,
                'scaler':    joblib.load(MODELS_DIR / f'{name}_scaler.pkl'),
                'kmer_dict': joblib.load(MODELS_DIR / f'{name}_kmer_dict.pkl'),
            }
        except Exception as e:
            print(f'  ✗ {pathogen}: {e}', file=sys.stderr)
    return models, preps


# ── features / prediction (identical to app.py) ─────────────────────────────────
def kmers(seq, k):
    total = len(seq) - k + 1
    if total <= 0:
        return Counter()
    c = Counter(seq[i:i+k] for i in range(total))
    return Counter({km: v / total for km, v in c.items()})


def seq_to_vector(sequence, k, kmer_dict):
    seq = sequence.upper().replace('U', 'T')
    counts = kmers(seq, k)
    return np.array([counts.get(km, 0) for km in sorted(kmer_dict)],
                    dtype=np.float32).reshape(1, -1)


def validate(raw):
    if not isinstance(raw, str) or len(raw) > MAX_SEQ_LEN * 2:
        return False, 'sequence too long or invalid'
    seq = raw.strip()
    if seq.startswith('>'):
        seq = ''.join(l for l in seq.splitlines() if not l.startswith('>'))
    seq = seq.upper().replace(' ', '').replace('\n', '').replace('\r', '')
    if not seq:
        return False, 'empty sequence'
    if len(seq) < 20:
        return False, 'sequence too short (min 20 nt)'
    if len(seq) > MAX_SEQ_LEN:
        return False, f'sequence too long (max {MAX_SEQ_LEN} nt)'
    invalid = set(seq) - set('ATCGURN')
    if invalid:
        return False, f"invalid characters: {', '.join(sorted(invalid))}"
    return True, seq


def get_prob(model, X):
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(X)[0][1])
    raw = float(model.predict(X)[0])
    return float(np.clip(1 / (1 + np.exp(-raw)), 0, 1))


def predict_one(seq, info, models, preps):
    """Return {pathogen: probability%} for a validated sequence."""
    out = {}
    for pathogen in info:
        if pathogen not in models:
            out[pathogen] = None
            continue
        try:
            prep = preps[pathogen]
            X = seq_to_vector(seq, info[pathogen]['k'], prep['kmer_dict'])
            X = prep['selector'].transform(X)
            if prep['pca'] is not None:
                X = prep['pca'].transform(X)
            X = prep['scaler'].transform(X)
            out[pathogen] = round(get_prob(models[pathogen], X) * 100, 2)
        except Exception:
            out[pathogen] = None
    return out


# ── input parsing ───────────────────────────────────────────────────────────────
def read_fasta(handle):
    """Minimal FASTA parser (id, sequence). Falls back for a single raw sequence."""
    seqid, buf = None, []
    for line in handle:
        line = line.rstrip('\n')
        if line.startswith('>'):
            if seqid is not None:
                yield seqid, ''.join(buf)
            seqid = line[1:].split()[0] or f'seq{sum(1 for _ in [0])}'
            buf = []
        elif line.strip():
            buf.append(line.strip())
    if seqid is not None:
        yield seqid, ''.join(buf)
    elif buf:  # no header at all: a bare sequence
        yield 'seq1', ''.join(buf)


def gather_inputs(args):
    if args.sequence:
        yield 'seq1', args.sequence
        return
    src = args.input
    if src in (None, '-'):
        yield from read_fasta(sys.stdin)
        return
    p = Path(src)
    if not p.exists():
        print(f'ERROR: input file not found: {src}', file=sys.stderr)
        sys.exit(2)
    with open(p) as fh:
        yield from read_fasta(fh)


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='BATATA CLI — predict lncRNA responsiveness to 12 potato biotic-stress pathogens.')
    ap.add_argument('input', nargs='?', default=None,
                    help="FASTA file, or '-' for stdin (omit if using --sequence)")
    ap.add_argument('--sequence', help='a single raw nucleotide sequence')
    ap.add_argument('-o', '--output', help='write CSV to this file (default: stdout)')
    ap.add_argument('--top', action='store_true',
                    help='print only each sequence id + its most likely pathogen')
    ap.add_argument('--min-prob', type=float, default=0.0,
                    help='with --top, only report if the top probability >= this %% (default 0)')
    args = ap.parse_args()

    info = load_config()
    models, preps = load_models(info)
    if not models:
        print('ERROR: no models loaded. Check models/saved/.', file=sys.stderr)
        sys.exit(1)
    print(f'  {len(models)}/{len(info)} models loaded', file=sys.stderr)

    out_fh = open(args.output, 'w', newline='') if args.output else sys.stdout
    writer = csv.writer(out_fh)
    if args.top:
        writer.writerow(['id', 'length', 'top_pathogen', 'probability_%'])
    else:
        writer.writerow(['id', 'length'] + PATHOGENS + ['top_pathogen', 'top_probability_%'])

    n_ok = n_skip = 0
    for seqid, raw in gather_inputs(args):
        ok, res = validate(raw)
        if not ok:
            print(f'  ⚠ {seqid}: skipped ({res})', file=sys.stderr)
            n_skip += 1
            continue
        seq = res
        probs = predict_one(seq, info, models, preps)
        valid = {p: v for p, v in probs.items() if v is not None}
        top = max(valid, key=valid.get) if valid else '—'
        topv = valid.get(top, 0.0)
        if args.top:
            if topv >= args.min_prob:
                writer.writerow([seqid, len(seq), top, topv])
        else:
            row = [seqid, len(seq)] + [probs.get(p, '') if probs.get(p) is not None else 'NA'
                                       for p in PATHOGENS] + [top, topv]
            writer.writerow(row)
        n_ok += 1

    if args.output:
        out_fh.close()
    print(f'  done: {n_ok} sequence(s) scored, {n_skip} skipped'
          + (f'  →  {args.output}' if args.output else ''), file=sys.stderr)


if __name__ == '__main__':
    main()
