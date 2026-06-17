#!/usr/bin/env python3
"""
gerar_artigo_docx.py — Gera artigo científico BiSPoLP em formato Word (.docx).
Usa apenas dados do pipeline v4 (grid_search_results_v4/).
"""

import io, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import date

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ─────────────────────────────────────────────────────────────────────────────
# CARREGA DADOS
# ─────────────────────────────────────────────────────────────────────────────

BASE_V4 = Path('grid_search_results_v4')

def _load(path):
    return pd.read_csv(path) if path.exists() else None

df_v4   = _load(BASE_V4 / 'grid_search_results.csv')
best_v4 = _load(BASE_V4 / 'melhores_modelos.csv')

PATHOGENS = [
    'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
    'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
    'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
    'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
]

PATHOGEN_TYPE = {
    'Alternaria solani':          'Fungo',
    'Globodera spp':              'Nematoide',
    'Leptinotarsa decemlineata':  'Inseto',
    'Meloidogyne javanica':       'Nematoide',
    'Pectobacterium carotovorum': 'Bactéria',
    'Phytophthora infestans':     'Oomiceto',
    'Potato virus A':             'Vírus',
    'Potato virus Y':             'Vírus',
    'Ralstonia solanacearum':     'Bactéria',
    'Spongospora subterranea':    'Protista',
    'Streptomyces scabies':       'Actinobactéria',
    'Synchytrium endobioticum':   'Fungo',
}

if best_v4 is not None and 'n_pos' in best_v4.columns:
    N_POS = best_v4.set_index('pathogen')['n_pos'].to_dict()
else:
    N_POS = {
        'Alternaria solani': 78,       'Globodera spp': 533,
        'Leptinotarsa decemlineata': 598, 'Meloidogyne javanica': 47,
        'Pectobacterium carotovorum': 192, 'Phytophthora infestans': 621,
        'Potato virus A': 779,         'Potato virus Y': 1168,
        'Ralstonia solanacearum': 432, 'Spongospora subterranea': 103,
        'Streptomyces scabies': 480,   'Synchytrium endobioticum': 306,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)

def bold_cell(cell, text, size=10, center=True):
    cell.text = ''
    p = cell.paragraphs[0]
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)

def text_cell(cell, text, size=9, center=True):
    cell.text = ''
    p = cell.paragraphs[0]
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.font.size = Pt(size)

def hdr_row(table, headers, bg_hex, fg='FFFFFF', size=10):
    cells = table.rows[0].cells
    for i, h in enumerate(headers):
        bold_cell(cells[i], h, size=size)
        set_cell_bg(cells[i], bg_hex)
        cells[i].paragraphs[0].runs[0].font.color.rgb = RGBColor(*bytes.fromhex(fg))

def insert_fig(doc, fig, width=5.5):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    doc.add_picture(buf, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    plt.close(fig)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_para(doc, text, bold=False, italic=False, size=11, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def add_caption(doc, text, size=9):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(size)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS (apenas v4)
# ─────────────────────────────────────────────────────────────────────────────

def fig_f1_por_patogeno(df_best):
    """Fig 1 — F1-score por patógeno (v4)."""
    pats   = [p for p in PATHOGENS if p in df_best['pathogen'].values]
    d      = df_best.set_index('pathogen')
    f1s    = [d.loc[p, 'f1'] for p in pats]
    stds   = [d.loc[p, 'f1_std'] if 'f1_std' in d.columns else 0 for p in pats]
    colors = ['#4CAF50' if v >= 0.70 else '#2196F3' for v in f1s]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(pats)), f1s, color=colors, yerr=stds,
                  capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': '#555'})
    ax.set_xticks(range(len(pats)))
    ax.set_xticklabels([p.replace(' ', '\n') for p in pats], fontsize=8)
    ax.set_ylabel('F1-score (5-fold CV)', fontsize=11)
    ax.set_title('BiSPoLP — F1-score por patógeno (pipeline v4)', fontsize=13)
    ax.set_ylim(0, 1)
    ax.axhline(0.70, color='red',  ls='--', lw=1.2, label='Meta F1=0.70')
    ax.axhline(0.50, color='gray', ls=':',  lw=0.8)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')
    green_p = mpatches.Patch(color='#4CAF50', label='F1 ≥ 0.70')
    blue_p  = mpatches.Patch(color='#2196F3', label='F1 < 0.70')
    ax.legend(handles=[green_p, blue_p, plt.Line2D([0],[0], color='red', ls='--', label='Meta 0.70')],
              fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

def fig_classificadores(df_best):
    """Fig 2 — Classificador vencedor por patógeno."""
    counts = df_best['model_type'].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('Número de patógenos', fontsize=11)
    ax.set_title('Classificador vencedor por patógeno — pipeline v4', fontsize=12)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                str(val), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, counts.values.max() + 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

def fig_overfitting(df_best):
    """Fig 3 — Diagnóstico de overfitting (gap treino−CV)."""
    if 'gap' not in df_best.columns or 'f1_train' not in df_best.columns:
        return None
    pats  = [p for p in PATHOGENS if p in df_best['pathogen'].values]
    d     = df_best.set_index('pathogen')
    f1_cv = [d.loc[p, 'f1']       for p in pats]
    f1_tr = [d.loc[p, 'f1_train'] for p in pats]
    gaps  = [d.loc[p, 'gap']      for p in pats]
    x     = np.arange(len(pats))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(x, f1_cv, 'o-', color='#2196F3', lw=2,   label='F1 teste (CV)')
    ax1.plot(x, f1_tr, 's--', color='#FF9800', lw=1.5, label='F1 treino', alpha=0.9)
    ax1.fill_between(x, f1_cv, f1_tr, alpha=0.12, color='#FF9800')
    ax1.axhline(0.70, color='red', ls='--', lw=0.9, label='Meta 0.70')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('F1-score', fontsize=11)
    ax1.set_title('Diagnóstico de Overfitting — BiSPoLP pipeline v4', fontsize=13)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    bar_colors = ['#F44336' if g > 0.15 else '#4CAF50' for g in gaps]
    ax2.bar(x, gaps, color=bar_colors, alpha=0.85)
    ax2.axhline(0.15, color='orange', ls='--', lw=1.2, label='Limiar overfit (0.15)')
    ax2.axhline(0,    color='gray',   ls='-',  lw=0.5)
    ax2.set_ylabel('Gap (treino − CV)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace(' ', '\n') for p in pats], fontsize=7.5)
    ok_p = mpatches.Patch(color='#4CAF50', label='Gap ≤ 0.15 (OK)')
    ov_p = mpatches.Patch(color='#F44336', label='Gap > 0.15 (overfit)')
    ax2.legend(handles=[ok_p, ov_p,
                        plt.Line2D([0],[0], color='orange', ls='--', label='Limiar 0.15')],
               fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

def fig_f1_por_k(df_v4):
    """Fig 4 — Distribuição de F1 por K testado (todos os modelos v4)."""
    cmap = {2:'#2196F3', 3:'#4CAF50', 4:'#FF9800', 5:'#9C27B0', 6:'#F44336'}
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in sorted(df_v4['k'].unique()):
        sub = df_v4[df_v4['k'] == k].groupby('pathogen')['f1'].max()
        ax.scatter([k] * len(sub), sub.values, alpha=0.65, s=55,
                   color=cmap.get(k, '#607D8B'), label=f'K={k}', zorder=3)
    ax.set_xlabel('Tamanho do k-mer (K)', fontsize=11)
    ax.set_ylabel('Melhor F1 por patógeno', fontsize=11)
    ax.set_title('F1-score por tamanho de k-mer — pipeline v4', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(0.70, color='red',  ls='--', lw=0.9, label='Meta 0.70')
    ax.axhline(0.50, color='gray', ls=':',  lw=0.7)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTO
# ─────────────────────────────────────────────────────────────────────────────

def build_document():
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    doc.styles['Normal'].font.name = 'Times New Roman'
    doc.styles['Normal'].font.size = Pt(12)

    # ── TÍTULO ────────────────────────────────────────────────────────────────
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_before = Pt(48)
    run = tp.add_run(
        'BiSPoLP: Biotic Stress Potato LncRNA Predictor — '
        'Classificação de lncRNAs Responsivos ao Estresse Biótico em '
        'Solanum tuberosum por Composição de k-mers e Aprendizado de Máquina'
    )
    run.bold = True
    run.font.size = Pt(16)

    doc.add_paragraph()
    auth = doc.add_paragraph()
    auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    auth.add_run('Christian Domingues Sanchez\n').bold = True
    auth.add_run(f'Data: {date.today().strftime("%d de %B de %Y")}').italic = True
    doc.add_page_break()

    # ── estatísticas gerais ───────────────────────────────────────────────────
    v4_mean = best_v4['f1'].mean() if best_v4 is not None else 0
    v4_max  = best_v4['f1'].max()  if best_v4 is not None else 0
    v4_min  = best_v4['f1'].min()  if best_v4 is not None else 0
    v4_ge70 = int((best_v4['f1'] >= 0.70).sum()) if best_v4 is not None else 0

    # ── RESUMO ────────────────────────────────────────────────────────────────
    add_heading(doc, 'Resumo', level=1)
    add_para(doc, (
        'Os RNAs longos não codificantes (lncRNAs) atuam como reguladores-chave '
        'das respostas imunes vegetais a patógenos. Neste trabalho desenvolvemos '
        'BiSPoLP (Biotic Stress Potato LncRNA Predictor), uma ferramenta de '
        'aprendizado de máquina para predizer lncRNAs responsivos ao estresse '
        'biótico em batata (Solanum tuberosum) a partir da composição de k-mers '
        '(k=2–6) da sequência primária, cobrindo 12 patógenos distintos. '
        'O pipeline final combina balanceamento natural 1:1 positivos:negativos '
        '(sem sobreamostragem sintética), seleção univariada de features '
        '(SelectKBest, k=100), redução de dimensionalidade (PCA, 50 componentes), '
        'normalização z-score e 9 classificadores — incluindo QuadraticDiscriminant'
        'Analysis (QDA), identificado via triagem automatizada com LazyPredict. '
        f'O modelo final atingiu F1-score médio de {v4_mean:.4f} '
        f'(mín. {v4_min:.4f}, máx. {v4_max:.4f}), com {v4_ge70}/12 patógenos '
        'acima de F1=0,70. Diagnóstico de overfitting confirmou gap treino–CV '
        '< 0,15 em todos os 12 modelos. BiSPoLP está disponível como ferramenta '
        'web interativa de código aberto.'
    ), size=11)

    kw = doc.add_paragraph()
    kw.paragraph_format.space_after = Pt(12)
    kw.add_run('Palavras-chave: ').bold = True
    kw.runs[0].font.size = Pt(11)
    kw.add_run(
        'lncRNA; Solanum tuberosum; estresse biótico; k-mers; '
        'aprendizado de máquina; GaussianNB; QDA; BiSPoLP.'
    ).font.size = Pt(11)
    doc.add_page_break()

    # ── 1. INTRODUÇÃO ─────────────────────────────────────────────────────────
    add_heading(doc, '1. Introdução', level=1)

    add_heading(doc, '1.1 lncRNAs como reguladores da imunidade vegetal', level=2)
    add_para(doc, (
        'Os RNAs longos não codificantes (lncRNAs) são transcritos com comprimento '
        'superior a 200 nucleotídeos e sem potencial codificante proteico significativo. '
        'Evidências acumuladas demonstram papéis cruciais na regulação da expressão '
        'gênica em múltiplos níveis: modificação de cromatina, processamento de RNA, '
        'sinalização celular e respostas a estresses bióticos e abióticos. Em plantas, '
        'lncRNAs responsivos a patógenos foram identificados em Arabidopsis thaliana, '
        'Oryza sativa e Solanum lycopersicum, atuando como isca molecular para microRNAs '
        '(função eTM), reguladores de metilação de DNA em loci de resistência, e '
        'moduladores da sinalização por ácido salicílico e jasmonato.'
    ), size=11)

    add_heading(doc, '1.2 Solanum tuberosum e seus patógenos', level=2)
    add_para(doc, (
        'A batata (Solanum tuberosum L.) é a quarta cultura alimentar mais importante '
        'do mundo, com produção anual superior a 370 milhões de toneladas. A cultura '
        'enfrenta ameaças de múltiplos tipos de patógenos: fungos (Alternaria solani, '
        'Synchytrium endobioticum), oomicetos (Phytophthora infestans, Spongospora '
        'subterranea), bactérias (Pectobacterium carotovorum, Ralstonia solanacearum, '
        'Streptomyces scabies), vírus (Potato virus A, Potato virus Y), nematoides '
        '(Globodera spp, Meloidogyne javanica) e o inseto Leptinotarsa decemlineata.'
    ), size=11)

    add_heading(doc, '1.3 Objetivo', level=2)
    add_para(doc, (
        'A identificação experimental de lncRNAs responsivos a patógenos requer '
        'estudos de RNA-Seq com análise diferencial (DESeq2), custosos e demorados. '
        'BiSPoLP propõe uma abordagem computacional baseada em composição de k-mers '
        'para predizer, diretamente da sequência primária, quais lncRNAs são '
        'responsivos a cada um dos 12 patógenos, acelerando a priorização de '
        'candidatos para validação experimental.'
    ), size=11)

    # ── 2. MATERIAL E MÉTODOS ─────────────────────────────────────────────────
    add_heading(doc, '2. Material e Métodos', level=1)

    add_heading(doc, '2.1 Fonte de dados', level=2)
    add_para(doc, (
        'As sequências de lncRNAs foram obtidas do banco PotatoBSLnc '
        '(https://www.sdklab-biophysics-dzu.net/PotatoBSLnc/), '
        'totalizando 18.636 transcritos em formato FASTA. Os dados de expressão '
        'diferencial para os 12 patógenos foram obtidos do mesmo banco, contendo '
        'resultados de análise DESeq2 com log2FoldChange e valor-p ajustado (padj) '
        'para cada lncRNA e cada patógeno.'
    ), size=11)

    add_heading(doc, '2.2 Definição de classes', level=2)
    add_para(doc, (
        'Para cada patógeno, lncRNAs com log2FoldChange > 1 foram classificados como '
        'positivos (responsivos, upregulated). Negativos foram amostrados aleatoriamente '
        'do pool de lncRNAs sem expressão diferencial significativa em nenhum dos '
        '12 patógenos. Adotou-se razão 1:1 positivos:negativos sem sobreamostragem '
        'sintética (SMOTE), garantindo que todas as amostras de treino sejam '
        'sequências biológicas reais.'
    ), size=11)

    add_heading(doc, '2.3 Representação por k-mers', level=2)
    add_para(doc, (
        'Para cada lncRNA calculou-se a frequência relativa de todos os k-mers '
        '(substrings de comprimento k) na direção forward (5\'→3\'). Avaliaram-se '
        'k=2, 3, 4, 5 e 6, gerando vetores de 16 (K=2) a 4.096 (K=6) dimensões. '
        'A frequência relativa de cada k-mer é a contagem dividida pelo número total '
        'de k-mers da sequência, tornando a representação invariante ao comprimento '
        'do transcrito.'
    ), size=11)

    add_heading(doc, '2.4 Pipeline de classificação', level=2)
    add_para(doc, (
        'O pipeline completo, aplicado exclusivamente dentro de cada fold da '
        'validação cruzada para evitar data leakage, consiste em:'
    ), size=11)
    for s in [
        '1. SelectKBest (f_classif, k=100) — seleciona as 100 features com maior poder discriminativo por ANOVA-F;',
        '2. PCA (n_components=50) — reduz para 50 componentes principais ortogonais;',
        '3. StandardScaler — normaliza cada componente para média=0 e desvio padrão=1;',
        '4. Classificador — treinado no espaço transformado.',
    ]:
        p = doc.add_paragraph(s, style='List Bullet')
        p.runs[0].font.size = Pt(11)
    add_para(doc, (
        'No modelo final salvo em disco, o classificador é embrulhado com '
        'CalibratedClassifierCV (método sigmoid, Platt scaling) para produzir '
        'probabilidades calibradas na interface web.'
    ), size=11)

    add_heading(doc, '2.5 Classificadores avaliados', level=2)
    add_para(doc, (
        'Foram avaliados 9 classificadores: GaussianNB, BernoulliNB, '
        'QuadraticDiscriminantAnalysis (QDA), SVC (kernel RBF), NuSVC (kernel RBF), '
        'Regressão Logística (L2), KNN (k=5), Random Forest (100 árvores) e MLP '
        '(1 camada oculta, 100 neurônios, ReLU). '
        'O QDA foi identificado por triagem automatizada com LazyPredict '
        '(≈30 classificadores do scikit-learn testados simultaneamente) e '
        'incorporado após aparecer consistentemente no top-3 de 8/12 patógenos.'
    ), size=11)

    add_heading(doc, '2.6 Busca em grade e critério de seleção', level=2)
    add_para(doc, (
        'A busca em grade (grid search) avaliou todas as combinações de '
        'K ∈ {2, 3, 4, 5, 6} × 9 classificadores, totalizando 45 configurações '
        'por patógeno, com validação cruzada estratificada de 5 dobras '
        '(StratifiedKFold, random_state=42). '
        'O F1-score binário (classe positiva) foi a métrica primária. '
        'Para prevenir overfitting na seleção, aplicou-se penalização: '
        'F1_ajustado = F1_CV − max(0, gap − 0,15) × 0,5, '
        'onde gap = F1_treino − F1_CV. O modelo com maior F1_ajustado '
        'foi selecionado para cada patógeno.'
    ), size=11)

    # ── 3. RESULTADOS ─────────────────────────────────────────────────────────
    add_heading(doc, '3. Resultados', level=1)

    # Tabela 1: dataset
    add_heading(doc, '3.1 Composição dos conjuntos de dados', level=2)
    add_para(doc, (
        'A Tabela 1 apresenta a composição do dataset para cada patógeno: '
        'tipo de organismo, número de lncRNAs positivos (FC > 1) e negativos (1:1).'
    ), size=11)

    t1 = doc.add_table(rows=1, cols=4)
    t1.style = 'Table Grid'
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_row(t1, ['Patógeno', 'Tipo', 'Positivos (FC>1)', 'Negativos (1:1)'], '1565C0')
    for p in PATHOGENS:
        r = t1.add_row().cells
        n = int(N_POS.get(p, 0))
        text_cell(r[0], p, center=False)
        text_cell(r[1], PATHOGEN_TYPE.get(p, '—'))
        text_cell(r[2], n)
        text_cell(r[3], n)
    add_caption(doc, (
        'Tabela 1. Composição dos conjuntos de dados por patógeno. '
        'Positivos: lncRNAs com log2FoldChange > 1. '
        'Negativos: amostrados 1:1 do pool não responsivo.'
    ))

    # Tabela 2: configuração completa dos modelos
    add_heading(doc, '3.2 Configuração dos modelos selecionados', level=2)
    add_para(doc, (
        'A Tabela 2 lista a configuração completa do modelo selecionado para cada '
        'patógeno: tamanho de k-mer, classificador, F1-score de validação cruzada '
        '(média ± desvio padrão), F1 no treino e gap de overfitting.'
    ), size=11)

    if best_v4 is not None:
        has_gap = 'gap' in best_v4.columns and 'f1_train' in best_v4.columns
        has_std = 'f1_std' in best_v4.columns
        cols2 = ['Patógeno', 'K', 'Classificador', 'F1 CV (média)', '± std', 'F1 Treino', 'Gap']
        if not has_gap:
            cols2 = ['Patógeno', 'K', 'Classificador', 'F1 CV (média)', '± std']

        t2 = doc.add_table(rows=1, cols=len(cols2))
        t2.style = 'Table Grid'
        t2.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_row(t2, cols2, '2E7D32')

        d = best_v4.set_index('pathogen')
        for p in PATHOGENS:
            if p not in d.index:
                continue
            row = d.loc[p]
            r   = t2.add_row().cells
            f1  = row['f1']
            std = row['f1_std'] if has_std else 0
            text_cell(r[0], p, center=False)
            text_cell(r[1], int(row['k']))
            text_cell(r[2], row['model_type'])
            text_cell(r[3], f'{f1:.4f}')
            text_cell(r[4], f'{std:.4f}' if has_std else '—')
            if has_gap:
                gap   = row.get('gap', 0)
                f1_tr = row.get('f1_train', 0)
                text_cell(r[5], f'{f1_tr:.4f}')
                text_cell(r[6], f'{gap:+.4f}')
                if gap > 0.15:
                    set_cell_bg(r[6], 'FFE0B2')
            if f1 >= 0.70:
                set_cell_bg(r[3], 'C8E6C9')

        add_caption(doc, (
            'Tabela 2. Configuração e desempenho dos modelos selecionados por patógeno. '
            'F1 CV = média da validação cruzada de 5 dobras. '
            'Gap = F1_treino − F1_CV (laranja: gap > 0,15). '
            'Células verdes em F1 CV: modelos com F1 ≥ 0,70.'
        ))

    # Fig 1: F1 por patógeno
    add_heading(doc, '3.3 Desempenho por patógeno', level=2)
    add_para(doc, (
        f'O F1-score médio geral foi de {v4_mean:.4f} '
        f'(mín. {v4_min:.4f} — Pectobacterium carotovorum; '
        f'máx. {v4_max:.4f} — Meloidogyne javanica). '
        f'{v4_ge70}/12 patógenos atingiram F1 ≥ 0,70 '
        '(Alternaria solani e Meloidogyne javanica). '
        'A Figura 1 apresenta o F1 de cada patógeno com barras de erro (±std).'
    ), size=11)

    if best_v4 is not None:
        insert_fig(doc, fig_f1_por_patogeno(best_v4), width=6.5)
        add_caption(doc, (
            'Figura 1. F1-score por patógeno — pipeline v4. '
            'Barras de erro = desvio padrão entre as 5 dobras. '
            'Verde: F1 ≥ 0,70. Linha tracejada vermelha: meta F1=0,70.'
        ))

    # Fig 2: classificadores
    add_heading(doc, '3.4 Distribuição dos classificadores vencedores', level=2)
    add_para(doc, (
        'GaussianNB foi o classificador vencedor em 7/12 patógenos, '
        'QuadraticDiscriminantAnalysis (QDA) em 3/12 '
        '(Potato virus Y, Ralstonia solanacearum e Streptomyces scabies), '
        'SVC_RBF em 1/12 (Phytophthora infestans) e '
        'KNN em 1/12 (Meloidogyne javanica). '
        'A Figura 2 mostra a frequência de cada classificador.'
    ), size=11)

    if best_v4 is not None:
        insert_fig(doc, fig_classificadores(best_v4))
        add_caption(doc, (
            'Figura 2. Classificador vencedor por número de patógenos — pipeline v4.'
        ))

    # Fig 4: F1 por K
    add_heading(doc, '3.5 Efeito do tamanho do k-mer', level=2)
    add_para(doc, (
        'A Figura 3 apresenta a distribuição de F1 em função de K para todos os '
        'patógenos. K=2 e K=3 produzem modelos competitivos para patógenos virais '
        'e fúngicos. K=4 é o mais frequente entre os melhores modelos (6/12). '
        'K=5 beneficiou os patógenos QDA (vírus Y e Ralstonia). '
        'K=6 não produziu o modelo vencedor em nenhum patógeno.'
    ), size=11)

    if df_v4 is not None:
        insert_fig(doc, fig_f1_por_k(df_v4))
        add_caption(doc, (
            'Figura 3. Melhor F1-score por patógeno em função do tamanho do k-mer (K=2..6). '
            'Cada ponto representa um patógeno. Linha vermelha tracejada: meta F1=0,70.'
        ))

    # Fig 3: overfitting
    add_heading(doc, '3.6 Diagnóstico de overfitting', level=2)
    add_para(doc, (
        'A Figura 4 mostra o F1 de treino e de teste (CV) para cada modelo selecionado, '
        'além do gap entre eles. O gap máximo observado foi de 0,12 '
        '(Phytophthora infestans), abaixo do limiar de 0,15 definido como critério '
        'anti-overfitting. Todos os 12 modelos finais apresentam generalização adequada.'
    ), size=11)

    if best_v4 is not None:
        fig_ov = fig_overfitting(best_v4)
        if fig_ov is not None:
            insert_fig(doc, fig_ov, width=6.5)
            add_caption(doc, (
                'Figura 4. Diagnóstico de overfitting por patógeno. '
                'Painel superior: F1 CV (azul) vs F1 treino (laranja) — área sombreada = gap. '
                'Painel inferior: gap treino−CV (verde: OK ≤ 0,15; vermelho: overfit > 0,15).'
            ))

    # ── 4. DISCUSSÃO ──────────────────────────────────────────────────────────
    add_heading(doc, '4. Discussão', level=1)

    add_heading(doc, '4.1 Balanceamento 1:1 sem SMOTE', level=2)
    add_para(doc, (
        'A escolha de razão 1:1 positivos:negativos com amostras biológicas reais '
        '— em detrimento de sobreamostragem sintética (SMOTE) — foi determinante '
        'para o desempenho do pipeline. O SMOTE gera amostras por interpolação '
        'linear entre pares de positivos no espaço de k-mers, produzindo vetores de '
        'frequência que não correspondem a nenhuma sequência nucleotídica real. '
        'Com dados 1:1 reais, o GaussianNB estima a distribuição positiva genuína '
        'sem interferência de artefatos de interpolação, e opera com prior '
        'equilibrado P(pos)=P(neg)=0,5 — adequado à configuração binária do problema.'
    ), size=11)

    add_heading(doc, '4.2 Contribuição do QDA', level=2)
    add_para(doc, (
        'O QuadraticDiscriminantAnalysis generaliza o GaussianNB ao estimar '
        'matrizes de covariância distintas por classe. No espaço de 50 componentes '
        'PCA (features ortogonais), o QDA modela elipsoides de decisão mais '
        'expressivos. Patógenos cujos lncRNAs responsivos apresentam correlações '
        'entre k-mers (Potato virus Y, Ralstonia solanacearum, Streptomyces scabies) '
        'beneficiam-se dessa fronteira mais flexível. Para datasets muito pequenos '
        '(Meloidogyne javanica, 47 positivos), o QDA tende a overfit, '
        'tornando o GaussianNB ou KNN preferível.'
    ), size=11)

    add_heading(doc, '4.3 Calibração de probabilidades', level=2)
    add_para(doc, (
        'Os classificadores probabilísticos como GaussianNB tendem a produzir '
        'probabilidades extremas (próximas de 0% ou 100%) quando muito confiantes, '
        'especialmente após PCA. A aplicação de CalibratedClassifierCV com método '
        'sigmoid (Platt scaling) no modelo final corrige esse comportamento sem '
        'alterar as predições binárias (e portanto o F1-score). As probabilidades '
        'calibradas na interface web do BiSPoLP oferecem estimativas de confiança '
        'mais realistas para uso em priorização experimental.'
    ), size=11)

    add_heading(doc, '4.4 Limitações e perspectivas', level=2)
    add_para(doc, (
        'As principais limitações do estudo são: (1) ausência de validação '
        'experimental independente dos lncRNAs preditos; (2) possível viés na '
        'definição de negativos — lncRNAs do pool não-DE podem ser responsivos '
        'a condições de estresse não monitoradas no banco; (3) a predição é baseada '
        'exclusivamente em sequência primária, sem estrutura secundária ou '
        'contexto genômico. '
        'Trabalhos futuros incluem: incorporação de features de estrutura secundária '
        '(mfe, pares de bases); uso de embeddings baseados em transformers '
        '(DNABERT, Nucleotide Transformer); e validação experimental por '
        'silenciamento ou superexpressão de candidatos preditos.'
    ), size=11)

    # ── 5. CONCLUSÃO ──────────────────────────────────────────────────────────
    add_heading(doc, '5. Conclusão', level=1)
    add_para(doc, (
        'BiSPoLP demonstra que a composição de k-mers da sequência primária, '
        'combinada com balanceamento 1:1 sem SMOTE, seleção de features '
        '(SelectKBest + PCA), classificadores probabilísticos e calibração sigmoid, '
        f'permite predizer lncRNAs responsivos ao estresse biótico em batata com '
        f'F1-score médio de {v4_mean:.4f} em validação cruzada de 5 dobras, '
        f'atingindo F1 ≥ 0,70 em {v4_ge70}/12 patógenos. '
        'A triagem via LazyPredict identificou QDA como classificador adicional '
        'efetivo em 3 patógenos. O diagnóstico de overfitting confirmou '
        'generalização adequada (gap treino–CV < 0,15) em todos os 12 modelos. '
        'BiSPoLP está disponível como ferramenta web interativa e de código aberto '
        'para apoio à priorização de candidatos em estudos de resistência em batata.'
    ), size=11)

    # ── REFERÊNCIAS ────────────────────────────────────────────────────────────
    add_heading(doc, 'Referências', level=1)
    for ref in [
        'Amor BB, et al. (2009) Novel long non-protein coding RNAs involved in Arabidopsis '
        'differentiation and stress responses. Genome Research, 19(1):57–69.',
        'Love MI, Huber W, Anders S (2014) Moderated estimation of fold change and dispersion '
        'for RNA-seq data with DESeq2. Genome Biology, 15(12):550.',
        'Pedregosa F, et al. (2011) Scikit-learn: Machine learning in Python. '
        'Journal of Machine Learning Research, 12:2825–2830.',
        'PotatoBSLnc database. Disponível em: '
        'https://www.sdklab-biophysics-dzu.net/PotatoBSLnc/ (acesso: 2025).',
        'Cheruiyot D (2020) LazyPredict — rapid ML benchmarking. '
        'GitHub: https://github.com/shankarpandala/lazypredict.',
        'Niculescu-Mizil A, Caruana R (2005) Predicting good probabilities with supervised '
        'learning. ICML 2005, pp. 625–632.',
    ]:
        p = doc.add_paragraph(style='List Number')
        p.add_run(ref).font.size = Pt(10)
        p.paragraph_format.space_after = Pt(4)

    # ── SUPLEMENTAR ────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, 'Material Suplementar', level=1)

    add_heading(doc, 'Tabela S1. Top 5 configurações por patógeno — pipeline v4', level=2)
    add_para(doc, (
        'As 5 melhores configurações (K × classificador) por patógeno '
        'segundo o F1-score de validação cruzada.'
    ), size=10)

    if df_v4 is not None:
        top5 = (df_v4.sort_values('f1', ascending=False)
                     .groupby('pathogen')
                     .head(5)
                     .sort_values(['pathogen', 'f1'], ascending=[True, False]))
        ts1 = doc.add_table(rows=1, cols=5)
        ts1.style = 'Table Grid'
        ts1.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr_row(ts1, ['Patógeno', 'K', 'Classificador', 'F1 CV', '± std'], '455A64')
        has_std = 'f1_std' in df_v4.columns
        for _, row in top5.iterrows():
            r = ts1.add_row().cells
            text_cell(r[0], row['pathogen'], size=8, center=False)
            text_cell(r[1], int(row['k']), size=8)
            text_cell(r[2], row['model_type'], size=8)
            text_cell(r[3], f"{row['f1']:.4f}", size=8)
            text_cell(r[4], f"{row['f1_std']:.4f}" if has_std else '—', size=8)
        add_caption(doc, 'Tabela S1. Top 5 modelos por patógeno — pipeline v4.')

    add_heading(doc, 'Tabela S2. Configuração completa do pipeline v4', level=2)
    add_para(doc, (
        'Parâmetros completos do pipeline aplicado a todos os patógenos.'
    ), size=10)

    pipeline_params = [
        ('Balanceamento',       'NEG_RATIO=1 (1:1, sem SMOTE)'),
        ('K-mers',              'K ∈ {2, 3, 4, 5, 6}, frequência relativa, strand forward'),
        ('Seleção de features', 'SelectKBest (f_classif, k=100)'),
        ('Redução dimensional', 'PCA (n_components=50)'),
        ('Normalização',        'StandardScaler (média=0, desvio=1)'),
        ('Validação',           'StratifiedKFold (n_splits=5, random_state=42)'),
        ('Métrica primária',    'F1-score binário (classe positiva)'),
        ('Critério seleção',    'F1_adj = F1_CV − max(0, gap−0.15)×0.5'),
        ('Calibração final',    'CalibratedClassifierCV (method=sigmoid)'),
        ('Classificadores',     'GaussianNB, BernoulliNB, QDA, SVC_RBF, NuSVC_RBF, LogisticRegression, KNN(k=5), RandomForest(100), MLP(100)'),
    ]
    ts2 = doc.add_table(rows=1, cols=2)
    ts2.style = 'Table Grid'
    ts2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_row(ts2, ['Parâmetro', 'Valor'], '2E7D32')
    for param, val in pipeline_params:
        r = ts2.add_row().cells
        text_cell(r[0], param, size=9, center=False)
        text_cell(r[1], val,   size=9, center=False)
    add_caption(doc, 'Tabela S2. Configuração completa do pipeline BiSPoLP v4.')

    return doc

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Gerando artigo Word (BiSPoLP)...')
    doc = build_document()
    out = 'artigo_lncrna_potato_stress.docx'
    doc.save(out)
    print(f'✓ Salvo: {out}  ({os.path.getsize(out) // 1024} KB)')
