#!/usr/bin/env python3
"""
Gera COMO_FUNCIONA.docx — documentação técnica do BiSPoLP
"""

import io
from pathlib import Path
from datetime import date
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── helpers ──────────────────────────────────────────────────────────────────

def heading(doc, text, level=1, color=None):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if color:
        for run in h.runs:
            run.font.color.rgb = RGBColor(*bytes.fromhex(color))
    return h

def para(doc, text, bold=False, italic=False, size=11, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p

def bullet(doc, items, size=11):
    for item in items:
        if isinstance(item, tuple):
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(item[0]).bold = True
            p.runs[0].font.size = Pt(size)
            p.add_run(item[1]).font.size = Pt(size)
        else:
            p = doc.add_paragraph(item, style='List Bullet')
            p.runs[0].font.size = Pt(size)

def numbered(doc, items, size=11):
    for item in items:
        p = doc.add_paragraph(item, style='List Number')
        p.runs[0].font.size = Pt(size)

def code_block(doc, lines, size=9.5):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after  = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(line)
        run.font.name = 'Courier New'
        run.font.size = Pt(size)

def table_2col(doc, rows_data, header=None, col_widths=(3.0, 3.5)):
    t = doc.add_table(rows=0, cols=2)
    t.style = 'Table Grid'
    if header:
        r = t.add_row().cells
        for i, h in enumerate(header):
            r[i].text = h
            r[i].paragraphs[0].runs[0].bold = True
            r[i].paragraphs[0].runs[0].font.size = Pt(10)
    for a, b in rows_data:
        r = t.add_row().cells
        r[0].text = a
        r[1].text = b
        for cell in r:
            cell.paragraphs[0].runs[0].font.size = Pt(10)
    doc.add_paragraph()
    return t

def divider(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run('─' * 80)
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

# ── conteúdo ─────────────────────────────────────────────────────────────────

def build():
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # ── CAPA ─────────────────────────────────────────────────────────────────
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_before = Pt(72)
    r = tp.add_run('BiSPoLP\nBiotic Stress Potato LncRNA Predictor')
    r.bold = True
    r.font.size = Pt(22)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run('Como funciona, como foi construído e por que as escolhas foram feitas assim').italic = True

    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f'Christian Domingues Sanchez — {date.today().strftime("%d/%m/%Y")}')

    doc.add_page_break()

    # ── ÍNDICE rápido ────────────────────────────────────────────────────────
    heading(doc, 'Índice', level=1)
    for item in [
        '1. O que é o BiSPoLP e para que serve',
        '2. De onde vêm os dados',
        '3. Como os lncRNAs são transformados em números (k-mers)',
        '4. Por que k-mers e não outra coisa',
        '5. O pipeline de machine learning — passo a passo',
        '6. Os classificadores testados e como escolhemos',
        '7. Overfitting e underfitting — o que é e como evitamos',
        '8. Por que NEG_RATIO=1 sem SMOTE foi a grande virada',
        '9. A descoberta do QDA via LazyPredict',
        '10. Calibração de probabilidades',
        '11. Resultados finais por patógeno',
        '12. Como rodar a ferramenta',
        '13. Como re-treinar os modelos',
        '14. Estrutura de arquivos do projeto',
    ]:
        p = doc.add_paragraph(item, style='List Bullet')
        p.runs[0].font.size = Pt(11)

    doc.add_page_break()

    # ── 1. O QUE É ───────────────────────────────────────────────────────────
    heading(doc, '1. O que é o BiSPoLP e para que serve', level=1)
    para(doc, (
        'BiSPoLP é uma ferramenta web de predição: você cola uma sequência de '
        'lncRNA (RNA longo não codificante) de batata, e ela diz qual dos '
        '12 patógenos testados tem mais chance de ativar aquele lncRNA durante '
        'a infecção, junto com a probabilidade estimada.'
    ))
    para(doc, (
        'Na prática, isso serve para priorizar experimentos. Em vez de fazer '
        'RNA-Seq para cada combinação lncRNA × patógeno (caro e demorado), você '
        'roda o BiSPoLP e foca os experimentos nos candidatos que o modelo apontou '
        'como mais prováveis.'
    ))
    para(doc, 'Os 12 patógenos cobertos:')
    table_2col(doc, [
        ('Alternaria solani',         'Fungo — pinta preta'),
        ('Globodera spp',             'Nematoide — nematoide de cisto'),
        ('Leptinotarsa decemlineata', 'Inseto — besouro da batata'),
        ('Meloidogyne javanica',      'Nematoide — nematoide das galhas'),
        ('Pectobacterium carotovorum','Bactéria — podridão mole'),
        ('Phytophthora infestans',    'Oomiceto — requeima'),
        ('Potato virus A',            'Vírus — PVA'),
        ('Potato virus Y',            'Vírus — PVY'),
        ('Ralstonia solanacearum',    'Bactéria — murcha bacteriana'),
        ('Spongospora subterranea',   'Protista — sarna pulverulenta'),
        ('Streptomyces scabies',      'Actinobactéria — sarna comum'),
        ('Synchytrium endobioticum',  'Fungo — verruga da batata'),
    ], header=['Patógeno', 'Tipo e doença'])

    # ── 2. DADOS ─────────────────────────────────────────────────────────────
    heading(doc, '2. De onde vêm os dados', level=1)
    para(doc, (
        'Todos os dados vêm do banco PotatoBSLnc '
        '(https://www.sdklab-biophysics-dzu.net/PotatoBSLnc/). '
        'O banco fornece:'
    ))
    bullet(doc, [
        ('lncRNA.fa: ', 'arquivo FASTA com as sequências de todos os lncRNAs de batata catalogados (~18 mil transcritos).'),
        ('Tabelas CSV por patógeno: ', 'resultados de análise de expressão diferencial (DESeq2) com colunas: '
         'lncRNA_id, log2FoldChange, padj (p-valor ajustado por FDR).'),
    ])
    para(doc, (
        'Para cada patógeno, separamos os lncRNAs em dois grupos:\n'
        '  Positivos: log2FoldChange > 1 (upregulated com o patógeno)\n'
        '  Negativos: lncRNAs que não foram significativos em nenhum patógeno'
    ))
    para(doc, (
        'Nota sobre o padj: no início do projeto tentamos usar o padj como peso '
        'nas amostras (lncRNAs com padj muito baixo = mais confiantes = peso maior). '
        'Isso não melhorou os resultados porque o SMOTE, que usávamos para balancear, '
        'cria amostras sintéticas por interpolação e apaga a informação de confiança. '
        'No pipeline final (v4) não usamos SMOTE nem pesos de padj — apenas o '
        'critério binário FC > 1 para definir positivos.'
    ), italic=True)

    # ── 3. K-MERS ────────────────────────────────────────────────────────────
    heading(doc, '3. Como os lncRNAs são transformados em números (k-mers)', level=1)
    para(doc, (
        'Um k-mer é uma subsequência de k nucleotídeos. Para K=2 (dímeros), '
        'os possíveis são: AA, AT, AC, AG, TA, TT, TC, TG, CA, CT, CC, CG, GA, GT, GC, GG '
        '(16 no total para K=2, 64 para K=3, 256 para K=4, 1024 para K=5, 4096 para K=6).'
    ))
    para(doc, 'Para cada lncRNA, contamos quantas vezes cada k-mer aparece na sequência e dividimos pelo total — isso dá a frequência relativa:')
    code_block(doc, [
        '  sequência:    ATGCAT',
        '  k-mers K=2:   AT, TG, GC, CA, AT  →  total=5',
        '  freq(AT) = 2/5 = 0.40',
        '  freq(TG) = 1/5 = 0.20',
        '  freq(GC) = 1/5 = 0.20',
        '  freq(CA) = 1/5 = 0.20',
        '  freq(demais) = 0.00',
    ])
    para(doc, (
        'O resultado é um vetor de comprimento 4^k. Para K=4 são 256 números por lncRNA. '
        'Esse vetor é a entrada do classificador. Testamos K de 2 a 6 — o melhor K '
        'é diferente para cada patógeno (buscamos via grid search).'
    ))

    # ── 4. POR QUE K-MERS ────────────────────────────────────────────────────
    heading(doc, '4. Por que k-mers e não outra coisa', level=1)
    para(doc, (
        'Vantagens dos k-mers para lncRNAs:\n'
        '• Não dependem de anotação — funcionam direto da sequência FASTA.\n'
        '• Capturam composição nucleotídica local que pode refletir motivos de '
        'ligação a proteínas regulatórias.\n'
        '• São rápidos de calcular (O(n) no comprimento da sequência).\n'
        '• Funcionam mesmo com sequências de comprimento variável (diferente de '
        'embeddings que precisam de padding).'
    ))
    para(doc, (
        'O que não testamos (trabalho futuro): estrutura secundária (folding), '
        'embeddings de transformers tipo DNABERT, e features de codon bias. '
        'Essas abordagens seriam computacionalmente mais pesadas e precisariam '
        'de mais dados para treinar.'
    ), italic=True)

    # ── 5. PIPELINE ML ───────────────────────────────────────────────────────
    heading(doc, '5. O pipeline de machine learning — passo a passo', level=1)
    para(doc, (
        'Cada patógeno tem seu próprio modelo treinado independentemente. '
        'O pipeline é igual para todos, mas com parâmetros diferentes (K, tipo de '
        'classificador) escolhidos por busca em grade (grid search) com validação '
        'cruzada de 5 dobras (5-fold stratified cross-validation).'
    ))
    para(doc, 'O pipeline dentro de cada dobra da CV:')
    numbered(doc, [
        'Montar dataset: positivos (FC > 1) + negativos amostrados 1:1',
        'SelectKBest(f_classif, k=100): seleciona as 100 features (k-mers) com '
        'maior estatística F de ANOVA entre positivos e negativos. '
        'Elimina ruído e reduz dimensão de 256–4096 para 100.',
        'PCA(n_components=50): redução adicional para 50 componentes principais, '
        'que captura a maior variância e deixa as features ortogonais (sem correlação entre si).',
        'StandardScaler: normaliza cada componente PCA para média=0 e desvio=1. '
        'Importante para SVM e KNN que são sensíveis a escala.',
        'Classificador: aprende a fronteira de decisão no espaço de 50 componentes.',
    ])
    para(doc, (
        'Importante: os passos 2, 3 e 4 são ajustados (fit) APENAS no conjunto de treino '
        'de cada dobra e depois aplicados (transform) no conjunto de teste. '
        'Isso evita vazamento de dados (data leakage), que inflaria artificialmente '
        'os scores.'
    ), italic=True)
    para(doc, 'No modelo final (que é salvo em pkl e usado na aplicação web):')
    numbered(doc, [
        'Todo o dataset do patógeno é usado (sem divisão treino/teste).',
        'O classificador bruto é embrulhado com CalibratedClassifierCV(method=sigmoid) '
        'para calibrar as probabilidades (veja seção 10).',
        'Os 5 objetos (selector, pca, scaler, kmer_dict, model) são salvos em disco.',
    ])

    # ── 6. CLASSIFICADORES ───────────────────────────────────────────────────
    heading(doc, '6. Os classificadores testados e como escolhemos', level=1)
    para(doc, 'Foram testados 9 classificadores no pipeline v4 (final):')
    table_2col(doc, [
        ('GaussianNB',              'Naive Bayes gaussiano. Assume features independentes com distribuição normal por classe. Rápido, interpretável, funciona bem com poucos dados.'),
        ('BernoulliNB',             'Naive Bayes para features binárias/esparsas. Menos adequado para frequências contínuas.'),
        ('QDA',                     'Análise Discriminante Quadrática. Estima covariâncias separadas por classe — fronteira de decisão mais flexível que GaussianNB.'),
        ('SVC (RBF)',               'Support Vector Machine com kernel radial. Boa margem de separação, mas sensível a escala e lento em datasets grandes.'),
        ('NuSVC (RBF)',             'Variante do SVC com parâmetro nu ao invés de C.'),
        ('Regressão Logística',     'Modelo linear com regularização L2. Interpreta combinação linear das features. Bom para fronteiras lineares no espaço PCA.'),
        ('KNN (k=5)',               'K vizinhos mais próximos. Sem assumir distribuição — puramente baseado em similaridade. Sensível a escala (por isso usamos StandardScaler).'),
        ('Random Forest (100)',     'Ensemble de 100 árvores de decisão. Robusto, mas com 50 features e dados pequenos tende a overfit.'),
        ('MLP (100 neurônios)',     'Rede neural rasa com uma camada oculta. Capacidade maior, mas requer mais dados e tempo.'),
    ], header=['Classificador', 'Como funciona e quando funciona melhor'])
    para(doc, (
        'A escolha do melhor classificador para cada patógeno é automática: '
        'o grid search testa todas as combinações (K × classificador) e escolhe '
        'a que maximiza o F1-score na validação cruzada, com penalização anti-overfitting '
        '(veja seção 7).'
    ))
    para(doc, (
        'O QDA foi descoberto depois, via LazyPredict — um pacote que testa '
        '~30 classificadores de uma vez. Veja a seção 9.'
    ), italic=True)

    # ── 7. OVERFITTING E UNDERFITTING ────────────────────────────────────────
    heading(doc, '7. Overfitting e underfitting — o que é e como evitamos', level=1)

    heading(doc, '7.1 O que é overfitting', level=2)
    para(doc, (
        'Overfitting ocorre quando o modelo "decora" o conjunto de treino em vez '
        'de aprender padrões generalizáveis. Sintoma: F1 no treino = 1.0, '
        'F1 no teste (CV) muito menor. O modelo funciona perfeitamente nos dados '
        'que viu, mas erra nos dados novos.'
    ))
    para(doc, 'Exemplo real que encontramos no projeto:')
    code_block(doc, [
        '  Meloidogyne javanica — SVC_RBF K=5',
        '  F1 treino = 1.0000   (perfeito — memorizou os 47 positivos)',
        '  F1 CV    = 0.7137   (bom, mas enganoso)',
        '  Gap      = 0.2863   → OVERFIT!',
    ])
    para(doc, (
        'Esse modelo parecia ótimo pelo F1 de CV, mas o gap enorme mostrava que ele '
        'estava memorizando. Com apenas 47 positivos, o SVC consegue separar '
        'perfeitamente os dados de treino — mas não generaliza.'
    ))

    heading(doc, '7.2 O que é underfitting', level=2)
    para(doc, (
        'Underfitting ocorre quando o modelo é simples demais para capturar os '
        'padrões dos dados. Sintoma: F1 baixo tanto no treino quanto no teste. '
        'No nosso caso, o pipeline v1 tinha underfitting generalizado '
        '(F1 médio ≈ 0.47) porque o SMOTE distorcia os dados de treino com '
        'amostras sintéticas.'
    ))

    heading(doc, '7.3 Como detectamos e corrigimos', level=2)
    para(doc, 'Implementamos 3 mecanismos:')
    bullet(doc, [
        ('Gap diagnóstico: ', 'dentro de cada dobra da CV, calculamos F1 no treino E no teste. '
         'gap = F1_treino − F1_CV. gap > 0.15 sinaliza overfitting.'),
        ('Penalização na seleção: ', 'ao escolher o melhor modelo por patógeno, usamos '
         'F1_ajustado = F1_CV − max(0, gap − 0.15) × 0.5. '
         'Isso penaliza modelos com gap alto, mesmo que o F1_CV pareça bom.'),
        ('Calibração (CalibratedClassifierCV): ', 'no modelo final, embrulhamos o classificador '
         'com calibração sigmoid. Isso não muda o F1, mas estabiliza as probabilidades.'),
    ])
    code_block(doc, [
        '  # Fórmula de penalização anti-overfitting',
        '  gap = f1_treino - f1_cv',
        '  if gap > 0.15:',
        '      f1_ajustado = f1_cv - (gap - 0.15) * 0.5',
        '  else:',
        '      f1_ajustado = f1_cv',
        '',
        '  # Calibração do modelo final',
        '  from sklearn.calibration import CalibratedClassifierCV',
        '  clf_final = CalibratedClassifierCV(raw_clf, method="sigmoid", cv=max(2, n_cv))',
        '  clf_final.fit(X_completo, y_completo)',
    ])
    para(doc, (
        'Resultado: todos os 12 modelos finais do pipeline v4 têm gap < 0.15. '
        'Para o Meloidogyne, o KNN K=2 (gap=0.06) foi escolhido no lugar do SVC_RBF K=5 (gap=0.28).'
    ))

    heading(doc, '7.4 Por que 5-fold CV e não holdout simples', level=2)
    para(doc, (
        'Com conjuntos pequenos (ex: Meloidogyne com 47 positivos), um único '
        'holdout de 20% teria apenas ~9 positivos no teste — muito ruidoso. '
        'A validação cruzada de 5 dobras usa 80% para treino e 20% para teste '
        'em cada rodada, e faz isso 5 vezes com divisões diferentes. '
        'O F1 final é a média das 5 rodadas — estimativa muito mais estável.'
    ))

    # ── 8. NEG_RATIO=1 SEM SMOTE ─────────────────────────────────────────────
    heading(doc, '8. Por que NEG_RATIO=1 sem SMOTE foi a grande virada', level=1)

    heading(doc, '8.1 Como era o pipeline v1 (baseline)', level=2)
    para(doc, (
        'O pipeline original (v1) usava:\n'
        '  NEG_RATIO=3: para cada positivo, 3 negativos.\n'
        '  SMOTE: gerava positivos sintéticos por interpolação até equilibrar as classes.\n'
        '  Resultado: F1 médio = 0.47 (variação 0.42–0.58).'
    ))

    heading(doc, '8.2 O problema do SMOTE com k-mers', level=2)
    para(doc, (
        'O SMOTE cria novas amostras interpolando pares de positivos reais no '
        'espaço de features. Por exemplo, com dois lncRNAs positivos A e B, '
        'o SMOTE cria C = A × 0.6 + B × 0.4. '
        'Mas C é um vetor de frequências de k-mers que não corresponde a '
        'nenhuma sequência biológica real — é um artefato matemático. '
        'Isso "borrou" a distribuição que o GaussianNB estava tentando aprender.'
    ))

    heading(doc, '8.3 A solução: 1:1 com amostras reais', level=2)
    para(doc, (
        'Com NEG_RATIO=1 (1 negativo por positivo) e sem SMOTE, '
        'todas as amostras são sequências biológicas reais. '
        'O GaussianNB consegue modelar a distribuição real de frequências '
        'de k-mers dos positivos sem interferência de dados sintéticos. '
        'O resultado foi um salto de +0.16 a +0.22 em F1 para todos os 12 patógenos.'
    ))
    table_2col(doc, [
        ('v1: NEG_RATIO=3 + SMOTE', 'F1 médio = 0.47'),
        ('v4: NEG_RATIO=1, sem SMOTE', 'F1 médio = 0.67 (+0.20)'),
    ], header=['Abordagem', 'Resultado'])

    # ── 9. QDA VIA LAZYPREDICT ───────────────────────────────────────────────
    heading(doc, '9. A descoberta do QDA via LazyPredict', level=1)
    para(doc, (
        'Depois de estabelecer o pipeline v4 com GaussianNB como dominante, '
        'rodamos o LazyPredict para ver se existiam outros classificadores '
        'ainda melhores. O LazyPredict testa ~30 classificadores do sklearn '
        'de uma vez, com a mesma CV honesta.'
    ))
    para(doc, 'Como rodamos (script: experiments/04_test_lazy.py):')
    code_block(doc, [
        '  from lazypredict.Supervised import LazyClassifier',
        '  clf = LazyClassifier(predictions=True)',
        '  models, preds = clf.fit(X_train_fold, X_test_fold, y_train, y_test)',
    ])
    para(doc, (
        'Resultado: QuadraticDiscriminantAnalysis (QDA) apareceu no top-3 de '
        '8/12 patógenos com K=4 fixo. Fizemos então um grid específico de '
        'QDA × K=2..6 para cada patógeno, e confirmamos ganho em 3 deles:'
    ))
    table_2col(doc, [
        ('Potato virus Y',       'GaussianNB K=4: 0.663 → QDA K=5: 0.675 (+0.012)'),
        ('Ralstonia solanacearum','GaussianNB K=4: 0.634 → QDA K=5: 0.681 (+0.047)'),
        ('Streptomyces scabies', 'GaussianNB K=3: 0.655 → QDA K=4: 0.676 (+0.021)'),
    ], header=['Patógeno', 'Antes → Depois (QDA)'])
    para(doc, (
        'Por que QDA funciona para esses 3? O QDA estima uma matriz de covariância '
        'por classe, ao contrário do GaussianNB que assume independência total. '
        'Para vírus e bactérias sistêmicas, o padrão de frequência de k-mers '
        'dos lncRNAs responsivos parece ter correlações entre posições que o '
        'QDA captura melhor. Para patógenos com poucos positivos (ex: Meloidogyne, '
        '47 amostras), o QDA overfita — GaussianNB é mais robusto.'
    ))

    # ── 10. CALIBRAÇÃO ───────────────────────────────────────────────────────
    heading(doc, '10. Calibração de probabilidades', level=1)
    para(doc, (
        'Antes da calibração, a ferramenta web às vezes mostrava probabilidades '
        'de 100% ou 0% para muitos patógenos de uma vez — o que é biologicamente '
        'implausível. Isso acontecia porque o GaussianNB, quando muito confiante, '
        'empurra as probabilidades para os extremos.'
    ))
    para(doc, (
        'Solução: CalibratedClassifierCV com método sigmoid (Platt scaling). '
        'Isso adiciona uma regressão logística em cima das saídas brutas do '
        'classificador para mapeá-las para probabilidades calibradas. '
        'É aplicado apenas no modelo final — não afeta a CV nem o F1.'
    ))
    code_block(doc, [
        '  from sklearn.calibration import CalibratedClassifierCV',
        '',
        '  raw_clf = GaussianNB()   # ou QDA, KNN, etc.',
        '  clf = CalibratedClassifierCV(raw_clf, method="sigmoid", cv=max(2, n_cv))',
        '  clf.fit(X_all, y_all)    # treina no dataset completo',
        '',
        '  # Agora predict_proba() retorna probabilidades calibradas',
        '  prob = clf.predict_proba(X_novo)[0][1]  # prob. da classe positiva',
    ])
    para(doc, 'Resultado: em vez de 0%/100%, as probabilidades ficaram distribuídas (ex: 33%, 51%, 28%), muito mais úteis para priorização.')

    # ── 11. RESULTADOS ───────────────────────────────────────────────────────
    heading(doc, '11. Resultados finais por patógeno (pipeline v4)', level=1)
    para(doc, (
        'Tabela completa de métricas para cada modelo — todas calculadas via '
        '5-fold cross-validation estratificada. '
        'MCC = Matthews Correlation Coefficient (varia de -1 a +1; '
        '0 = predição aleatória, 1 = perfeito).'
    ))

    # Tabela completa de métricas
    import json
    from pathlib import Path as _Path
    cfg_path = _Path(__file__).parent.parent / 'models' / 'best_models_config.json'
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    pathogens_order = [
        'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
        'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
        'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
        'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
    ]
    model_map = {
        'Alternaria solani': ('K=3','GaussianNB'),
        'Globodera spp': ('K=4','GaussianNB'),
        'Leptinotarsa decemlineata': ('K=4','GaussianNB'),
        'Meloidogyne javanica': ('K=2','KNN'),
        'Pectobacterium carotovorum': ('K=4','Logistic Reg.'),
        'Phytophthora infestans': ('K=2','SVC_RBF'),
        'Potato virus A': ('K=2','GaussianNB'),
        'Potato virus Y': ('K=5','QDA'),
        'Ralstonia solanacearum': ('K=5','QDA'),
        'Spongospora subterranea': ('K=4','GaussianNB'),
        'Streptomyces scabies': ('K=4','QDA'),
        'Synchytrium endobioticum': ('K=3','GaussianNB'),
    }

    from docx import Document as _Doc
    from docx.shared import Pt as _Pt
    from docx.enum.table import WD_TABLE_ALIGNMENT as _TA
    from docx.enum.text import WD_ALIGN_PARAGRAPH as _WA
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _OE
    from docx.shared import RGBColor as _RGB

    def _bg(cell, hex_c):
        tc = cell._tc; p = tc.get_or_add_tcPr()
        s = _OE('w:shd'); s.set(_qn('w:val'),'clear')
        s.set(_qn('w:color'),'auto'); s.set(_qn('w:fill'), hex_c)
        p.append(s)

    t = doc.add_table(rows=1, cols=8)
    t.style = 'Table Grid'
    t.alignment = _TA.CENTER
    headers = ['Patógeno', 'K / Modelo', 'F1', 'Precisão', 'Recall', 'ROC-AUC', 'Acurácia', 'MCC']
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ''
        run = c.paragraphs[0].add_run(h)
        run.bold = True; run.font.size = _Pt(9)
        c.paragraphs[0].alignment = _WA.CENTER
        _bg(c, '1565C0')
        run.font.color.rgb = _RGB(0xFF,0xFF,0xFF)

    for p_name in pathogens_order:
        m   = cfg.get(p_name, {})
        k_m = model_map.get(p_name, ('—','—'))
        row = t.add_row().cells
        for ci, val in enumerate([
            p_name,
            f'{k_m[0]}  {k_m[1]}',
            f'{m.get("f1",0):.4f}',
            f'{m.get("precision",0):.4f}',
            f'{m.get("recall",0):.4f}',
            f'{m.get("roc_auc",0):.4f}',
            f'{m.get("accuracy",0):.4f}',
            f'{m.get("mcc",0):.4f}',
        ]):
            row[ci].text = ''
            run = row[ci].paragraphs[0].add_run(val)
            run.font.size = _Pt(8.5)
            row[ci].paragraphs[0].alignment = _WA.LEFT if ci < 2 else _WA.CENTER
        if m.get('f1', 0) >= 0.70:
            _bg(row[2], 'C8E6C9')
        if m.get('roc_auc', 0) >= 0.70:
            _bg(row[5], 'C8E6C9')

    doc.add_paragraph()

    # Legenda das métricas
    para(doc, 'O que significa cada métrica:', bold=True)
    bullet(doc, [
        ('F1-score: ',     'média harmônica de precisão e recall. Equilibra os dois — boa para classes desbalanceadas.'),
        ('Precisão: ',     'dos lncRNAs que o modelo disse "responsivo", quantos realmente são? Alta precisão = menos falsos positivos.'),
        ('Recall: ',       'dos lncRNAs que são responsivos, quantos o modelo detectou? Alto recall = menos falsos negativos.'),
        ('ROC-AUC: ',      'área sob a curva ROC. Mede separabilidade das classes independente do limiar. 0.5 = aleatório, 1.0 = perfeito.'),
        ('Acurácia: ',     'proporção total de predições corretas (positivos + negativos).'),
        ('MCC: ',          'Matthews Correlation Coefficient. Métrica robusta que considera todos os quadrantes da matriz de confusão. Varia de -1 a +1.'),
    ])

    # ── 12. COMO RODAR ───────────────────────────────────────────────────────
    heading(doc, '12. Como rodar a ferramenta', level=1)
    para(doc, 'Requisitos: Python 3.10+, instale as dependências com:')
    code_block(doc, ['  pip install -r requirements.txt'])
    para(doc, 'Iniciar a aplicação web:')
    code_block(doc, [
        '  # Linux/Mac',
        '  ./run.sh',
        '',
        '  # Windows',
        '  run.bat',
        '',
        '  # Ou diretamente:',
        '  python app.py',
    ])
    para(doc, 'Acesse http://localhost:5000 no navegador.')
    para(doc, (
        'Se o banco de sequências (lncRNA.fa) estiver em outro local, defina a '
        'variável de ambiente BISPOLP_DATA_PATH:'
    ))
    code_block(doc, [
        '  BISPOLP_DATA_PATH=/meu/caminho/para/potato_data python app.py',
    ])
    para(doc, (
        'Sem o FASTA, a ferramenta funciona normalmente para predição — '
        'apenas o botão "sequência aleatória" fica desabilitado.'
    ))

    # ── 13. COMO RE-TREINAR ──────────────────────────────────────────────────
    heading(doc, '13. Como re-treinar os modelos', level=1)
    para(doc, (
        'O script de treino principal é src/train.py. '
        'Ele usa os dados brutos do banco PotatoBSLnc para re-treinar '
        'todos os 12 modelos do zero.'
    ))
    code_block(doc, [
        '  # Rodar do diretório BiSPoLP-ML/',
        '  python src/train.py',
        '',
        '  # Os modelos serão salvos em models/saved/',
        '  # Os resultados do grid search em results/v4/',
        '  # O config JSON será atualizado em models/best_models_config.json',
    ])
    para(doc, 'Tempo estimado: 30–60 minutos no total, dependendo do hardware.')
    para(doc, 'Os experimentos que levaram ao pipeline v4 estão em experiments/:')
    table_2col(doc, [
        ('01_baseline_v1.py',    'Pipeline original (v1) com NEG_RATIO=3 + SMOTE — referência baseline.'),
        ('02_test_abordagens.py','Testa 8 abordagens diferentes sistematicamente para tentar melhorar o F1.'),
        ('03_test_fwd_rev.py',   'Experimenta k-mers forward + reverse complement — não melhorou.'),
        ('04_test_lazy.py',      'LazyPredict: triagem de ~30 classificadores + grid QDA × K=2..6.'),
    ], header=['Script', 'O que faz'])

    # ── 14. ESTRUTURA DE ARQUIVOS ─────────────────────────────────────────────
    heading(doc, '14. Estrutura de arquivos do projeto', level=1)
    code_block(doc, [
        'BiSPoLP-ML/',
        '├── app.py                  ← Aplicação web Flask (rode este)',
        '├── run.sh / run.bat        ← Atalhos para iniciar',
        '├── requirements.txt        ← Dependências Python',
        '├── .gitignore',
        '│',
        '├── templates/',
        '│   └── index.html          ← Interface web (EN/PT-BR)',
        '│',
        '├── models/',
        '│   ├── best_models_config.json  ← Config: K, tipo, F1 por patógeno',
        '│   └── saved/              ← Modelos treinados (.pkl)',
        '│       ├── Alternaria_solani_model.pkl',
        '│       ├── Alternaria_solani_selector.pkl',
        '│       ├── Alternaria_solani_pca.pkl',
        '│       ├── Alternaria_solani_scaler.pkl',
        '│       ├── Alternaria_solani_kmer_dict.pkl',
        '│       └── ... (5 arquivos × 12 patógenos = 60 arquivos)',
        '│',
        '├── src/',
        '│   ├── train.py            ← Script de (re)treino completo',
        '│   └── predict.py          ← Versão alternativa da app',
        '│',
        '├── experiments/',
        '│   ├── 01_baseline_v1.py',
        '│   ├── 02_test_abordagens.py',
        '│   ├── 03_test_fwd_rev.py',
        '│   ├── 04_test_lazy.py',
        '│   └── results/            ← CSVs dos experimentos',
        '│',
        '├── results/',
        '│   ├── v1/                 ← grid_search_results.csv, melhores_modelos.csv',
        '│   └── v4/                 ← grid_search_results.csv, melhores_modelos.csv',
        '│',
        '├── article/',
        '│   ├── generate_article.py ← Gera o artigo .docx',
        '│   └── artigo_lncrna_potato_stress.docx',
        '│',
        '└── docs/',
        '    ├── gerar_documentacao.py  ← Este script',
        '    └── COMO_FUNCIONA.docx     ← Esta documentação',
    ])

    return doc


def build_en():
    """English version of the documentation."""
    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # ── COVER ────────────────────────────────────────────────────────────────
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_before = Pt(72)
    r = tp.add_run('BiSPoLP\nBiotic Stress Potato LncRNA Predictor')
    r.bold = True; r.font.size = Pt(22)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run('How it works, how it was built, and why the choices were made').italic = True
    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run(f'Christian Domingues Sanchez — {date.today().strftime("%B %d, %Y")}')
    doc.add_page_break()

    # ── TABLE OF CONTENTS ────────────────────────────────────────────────────
    heading(doc, 'Table of Contents', level=1)
    for item in [
        '1. What is BiSPoLP and what is it for',
        '2. Where the data comes from',
        '3. How lncRNAs are converted into numbers (k-mers)',
        '4. Why k-mers and not something else',
        '5. The machine learning pipeline — step by step',
        '6. Classifiers tested and how we chose them',
        '7. Overfitting and underfitting — what they are and how we avoided them',
        '8. Why NEG_RATIO=1 without SMOTE was the turning point',
        '9. Discovering QDA via LazyPredict',
        '10. Probability calibration',
        '11. Final results per pathogen',
        '12. How to run the tool',
        '13. How to retrain the models',
        '14. Project file structure',
    ]:
        p = doc.add_paragraph(item, style='List Bullet')
        p.runs[0].font.size = Pt(11)
    doc.add_page_break()

    # ── 1. WHAT IS BISPOLP ───────────────────────────────────────────────────
    heading(doc, '1. What is BiSPoLP and what is it for', level=1)
    para(doc, (
        'BiSPoLP is a web prediction tool: you paste a potato lncRNA sequence '
        '(long non-coding RNA), and it tells you which of the 12 tested pathogens '
        'is most likely to activate that lncRNA during infection, '
        'along with the estimated probability.'
    ))
    para(doc, (
        'In practice, this is useful for prioritising experiments. Instead of running '
        'RNA-Seq for every lncRNA × pathogen combination (expensive and time-consuming), '
        'you run BiSPoLP and focus experiments on the candidates the model flagged '
        'as most likely.'
    ))
    para(doc, 'The 12 pathogens covered:')
    table_2col(doc, [
        ('Alternaria solani',         'Fungus — early blight'),
        ('Globodera spp',             'Nematode — potato cyst nematode'),
        ('Leptinotarsa decemlineata', 'Insect — Colorado potato beetle'),
        ('Meloidogyne javanica',      'Nematode — root-knot nematode'),
        ('Pectobacterium carotovorum','Bacterium — soft rot'),
        ('Phytophthora infestans',    'Oomycete — late blight'),
        ('Potato virus A',            'Virus — PVA'),
        ('Potato virus Y',            'Virus — PVY'),
        ('Ralstonia solanacearum',    'Bacterium — bacterial wilt'),
        ('Spongospora subterranea',   'Protist — powdery scab'),
        ('Streptomyces scabies',      'Actinobacterium — common scab'),
        ('Synchytrium endobioticum',  'Fungus — potato wart'),
    ], header=['Pathogen', 'Type and disease'])

    # ── 2. DATA ──────────────────────────────────────────────────────────────
    heading(doc, '2. Where the data comes from', level=1)
    para(doc, (
        'All data come from the PotatoBSLnc database '
        '(https://www.sdklab-biophysics-dzu.net/PotatoBSLnc/). '
        'The database provides:'
    ))
    bullet(doc, [
        ('lncRNA.fa: ', 'FASTA file with sequences of all catalogued potato lncRNAs (~18,000 transcripts).'),
        ('CSV tables per pathogen: ', 'results of differential expression analysis (DESeq2) with columns: '
         'lncRNA_id, log2FoldChange, padj (FDR-adjusted p-value).'),
    ])
    para(doc, (
        'For each pathogen, we split lncRNAs into two groups:\n'
        '  Positives: log2FoldChange > 1 (upregulated with the pathogen)\n'
        '  Negatives: lncRNAs not significant in any pathogen'
    ))
    para(doc, (
        'Note on padj: early in the project we tried using padj as sample weights '
        '(lncRNAs with very low padj = more confident = higher weight). '
        'This did not improve results because SMOTE, which we used to balance classes, '
        'creates synthetic samples by interpolation and destroys the confidence signal. '
        'In the final pipeline (v4) we use neither SMOTE nor padj weights — '
        'only the binary FC > 1 criterion to define positives.'
    ), italic=True)

    # ── 3. K-MERS ────────────────────────────────────────────────────────────
    heading(doc, '3. How lncRNAs are converted into numbers (k-mers)', level=1)
    para(doc, (
        'A k-mer is a substring of k nucleotides. For K=2 (dimers), '
        'the possible ones are: AA, AT, AC, AG, TA, TT, TC, TG, CA, CT, CC, CG, GA, GT, GC, GG '
        '(16 total for K=2, 64 for K=3, 256 for K=4, 1024 for K=5, 4096 for K=6).'
    ))
    para(doc, 'For each lncRNA, we count how many times each k-mer appears and divide by the total — this gives the relative frequency:')
    code_block(doc, [
        '  sequence:     ATGCAT',
        '  k-mers K=2:   AT, TG, GC, CA, AT  →  total=5',
        '  freq(AT) = 2/5 = 0.40',
        '  freq(TG) = 1/5 = 0.20',
        '  freq(GC) = 1/5 = 0.20',
        '  freq(CA) = 1/5 = 0.20',
        '  freq(others) = 0.00',
    ])
    para(doc, (
        'The result is a vector of length 4^k. For K=4 that is 256 numbers per lncRNA. '
        'This vector is the classifier input. We tested K from 2 to 6 — the best K '
        'differs per pathogen (found via grid search).'
    ))

    # ── 4. WHY K-MERS ────────────────────────────────────────────────────────
    heading(doc, '4. Why k-mers and not something else', level=1)
    para(doc, (
        'Advantages of k-mers for lncRNAs:\n'
        '• No annotation required — work directly from FASTA sequence.\n'
        '• Capture local nucleotide composition that may reflect binding motifs '
        'for regulatory proteins.\n'
        '• Fast to compute (O(n) in sequence length).\n'
        '• Work with variable-length sequences (unlike embeddings that require padding).'
    ))
    para(doc, (
        'What we did not test (future work): secondary structure (RNA folding), '
        'transformer-based embeddings such as DNABERT, and codon bias features. '
        'These approaches would be computationally heavier and would require '
        'more data to train.'
    ), italic=True)

    # ── 5. ML PIPELINE ───────────────────────────────────────────────────────
    heading(doc, '5. The machine learning pipeline — step by step', level=1)
    para(doc, (
        'Each pathogen has its own independently trained model. '
        'The pipeline is the same for all, but with different parameters (K, classifier type) '
        'chosen by grid search with 5-fold stratified cross-validation.'
    ))
    para(doc, 'The pipeline inside each CV fold:')
    numbered(doc, [
        'Build dataset: positives (FC > 1) + negatives sampled 1:1',
        'SelectKBest(f_classif, k=100): selects the 100 features (k-mers) with '
        'the highest ANOVA F-statistic between positives and negatives. '
        'Removes noise and reduces dimensionality from 256–4096 to 100.',
        'PCA(n_components=50): further reduction to 50 principal components, '
        'capturing the most variance and making features orthogonal (uncorrelated).',
        'StandardScaler: normalises each PCA component to mean=0 and std=1. '
        'Important for SVM and KNN which are scale-sensitive.',
        'Classifier: learns the decision boundary in the 50-component space.',
    ])
    para(doc, (
        'Important: steps 2, 3, and 4 are fitted ONLY on the training set of each fold '
        'and then applied (transform) to the test set. '
        'This prevents data leakage, which would artificially inflate scores.'
    ), italic=True)
    para(doc, 'In the final model (saved as pkl and used in the web app):')
    numbered(doc, [
        'The full dataset of the pathogen is used (no train/test split).',
        'The raw classifier is wrapped with CalibratedClassifierCV(method=sigmoid) '
        'to calibrate probabilities (see section 10).',
        'The 5 objects (selector, pca, scaler, kmer_dict, model) are saved to disk.',
    ])

    # ── 6. CLASSIFIERS ───────────────────────────────────────────────────────
    heading(doc, '6. Classifiers tested and how we chose them', level=1)
    para(doc, '9 classifiers were tested in pipeline v4 (final):')
    table_2col(doc, [
        ('GaussianNB',           'Gaussian Naive Bayes. Assumes independent features with normal distribution per class. Fast, interpretable, works well with little data.'),
        ('BernoulliNB',          'Naive Bayes for binary/sparse features. Less suited for continuous frequencies.'),
        ('QDA',                  'Quadratic Discriminant Analysis. Estimates separate covariance matrices per class — more flexible decision boundary than GaussianNB.'),
        ('SVC (RBF)',             'Support Vector Machine with radial kernel. Good separation margin, but scale-sensitive and slow on large datasets.'),
        ('NuSVC (RBF)',           'SVC variant with nu parameter instead of C.'),
        ('Logistic Regression',  'Linear model with L2 regularisation. Interprets linear combination of features. Good for linear boundaries in PCA space.'),
        ('KNN (k=5)',             'K nearest neighbours. No distributional assumption — purely similarity-based. Scale-sensitive (hence StandardScaler).'),
        ('Random Forest (100)',  'Ensemble of 100 decision trees. Robust, but with 50 features and small data it tends to overfit.'),
        ('MLP (100 neurons)',    'Shallow neural network with one hidden layer. More capacity, but needs more data and training time.'),
    ], header=['Classifier', 'How it works and when it performs best'])
    para(doc, (
        'The best classifier for each pathogen is selected automatically: '
        'grid search tests all combinations (K × classifier) and picks '
        'the one maximising CV F1-score, with anti-overfitting penalty (see section 7).'
    ))
    para(doc, (
        'QDA was discovered later, via LazyPredict — a package that tests '
        '~30 classifiers at once. See section 9.'
    ), italic=True)

    # ── 7. OVERFITTING ───────────────────────────────────────────────────────
    heading(doc, '7. Overfitting and underfitting — what they are and how we avoided them', level=1)
    heading(doc, '7.1 What is overfitting', level=2)
    para(doc, (
        'Overfitting occurs when the model "memorises" the training set instead '
        'of learning generalisable patterns. Symptom: F1 on training = 1.0, '
        'F1 on test (CV) much lower. The model works perfectly on data it has seen, '
        'but fails on new data.'
    ))
    para(doc, 'Real example from this project:')
    code_block(doc, [
        '  Meloidogyne javanica — SVC_RBF K=5',
        '  Train F1 = 1.0000   (perfect — memorised the 47 positives)',
        '  CV F1    = 0.7137   (good, but misleading)',
        '  Gap      = 0.2863   → OVERFIT!',
    ])
    para(doc, (
        'This model seemed great by CV F1, but the huge gap showed it was memorising. '
        'With only 47 positives, SVC can perfectly separate the training data — '
        'but does not generalise.'
    ))
    heading(doc, '7.2 What is underfitting', level=2)
    para(doc, (
        'Underfitting occurs when the model is too simple to capture the data patterns. '
        'Symptom: low F1 on both training and test. '
        'In our case, pipeline v1 had generalised underfitting '
        '(mean F1 ≈ 0.47) because SMOTE was distorting the training data '
        'with synthetic samples.'
    ))
    heading(doc, '7.3 How we detected and corrected it', level=2)
    para(doc, 'We implemented 3 mechanisms:')
    bullet(doc, [
        ('Gap diagnostic: ', 'within each CV fold, we compute F1 on training AND test. '
         'gap = F1_train − F1_CV. gap > 0.15 signals overfitting.'),
        ('Selection penalty: ', 'when selecting the best model per pathogen, we use '
         'F1_adjusted = F1_CV − max(0, gap − 0.15) × 0.5. '
         'This penalises models with high gap, even if CV F1 looks good.'),
        ('Calibration (CalibratedClassifierCV): ', 'in the final model, we wrap the classifier '
         'with sigmoid calibration. This does not change F1, but stabilises probabilities.'),
    ])
    code_block(doc, [
        '  # Anti-overfitting penalty formula',
        '  gap = f1_train - f1_cv',
        '  if gap > 0.15:',
        '      f1_adjusted = f1_cv - (gap - 0.15) * 0.5',
        '  else:',
        '      f1_adjusted = f1_cv',
        '',
        '  # Final model calibration',
        '  from sklearn.calibration import CalibratedClassifierCV',
        '  clf_final = CalibratedClassifierCV(raw_clf, method="sigmoid", cv=max(2, n_cv))',
        '  clf_final.fit(X_full, y_full)',
    ])
    para(doc, (
        'Result: all 12 final pipeline v4 models have gap < 0.15. '
        'For Meloidogyne, KNN K=2 (gap=0.06) was chosen over SVC_RBF K=5 (gap=0.28).'
    ))
    heading(doc, '7.4 Why 5-fold CV and not simple holdout', level=2)
    para(doc, (
        'With small datasets (e.g. Meloidogyne with 47 positives), a single '
        '20% holdout would have only ~9 positives in the test — very noisy. '
        '5-fold cross-validation uses 80% for training and 20% for testing '
        'in each round, doing this 5 times with different splits. '
        'The final F1 is the mean of the 5 rounds — a much more stable estimate.'
    ))

    # ── 8. NEG_RATIO=1 ───────────────────────────────────────────────────────
    heading(doc, '8. Why NEG_RATIO=1 without SMOTE was the turning point', level=1)
    heading(doc, '8.1 What pipeline v1 (baseline) looked like', level=2)
    para(doc, (
        'The original pipeline (v1) used:\n'
        '  NEG_RATIO=3: for each positive, 3 negatives.\n'
        '  SMOTE: generated synthetic positives by interpolation to balance classes.\n'
        '  Result: mean F1 = 0.47 (range 0.42–0.58).'
    ))
    heading(doc, '8.2 The problem of SMOTE with k-mers', level=2)
    para(doc, (
        'SMOTE creates new samples by interpolating pairs of real positives in '
        'feature space. For example, with two positive lncRNAs A and B, '
        'SMOTE creates C = A × 0.6 + B × 0.4. '
        'But C is a k-mer frequency vector that corresponds to no real biological '
        'sequence — it is a mathematical artefact. '
        'This "blurred" the distribution that GaussianNB was trying to learn.'
    ))
    heading(doc, '8.3 The solution: 1:1 with real samples', level=2)
    para(doc, (
        'With NEG_RATIO=1 (1 negative per positive) and no SMOTE, '
        'all samples are real biological sequences. '
        'GaussianNB can model the real distribution of k-mer frequencies '
        'of positives without interference from synthetic data. '
        'The result was a jump of +0.16 to +0.22 in F1 for all 12 pathogens.'
    ))
    table_2col(doc, [
        ('v1: NEG_RATIO=3 + SMOTE', 'Mean F1 = 0.47'),
        ('v4: NEG_RATIO=1, no SMOTE', 'Mean F1 = 0.67 (+0.20)'),
    ], header=['Approach', 'Result'])

    # ── 9. QDA VIA LAZYPREDICT ───────────────────────────────────────────────
    heading(doc, '9. Discovering QDA via LazyPredict', level=1)
    para(doc, (
        'After establishing pipeline v4 with GaussianNB as dominant, '
        'we ran LazyPredict to check for even better classifiers. '
        'LazyPredict tests ~30 sklearn classifiers at once, '
        'with the same honest CV.'
    ))
    para(doc, 'How we ran it (script: experiments/04_test_lazy.py):')
    code_block(doc, [
        '  from lazypredict.Supervised import LazyClassifier',
        '  clf = LazyClassifier(predictions=True)',
        '  models, preds = clf.fit(X_train_fold, X_test_fold, y_train, y_test)',
    ])
    para(doc, (
        'Result: QuadraticDiscriminantAnalysis (QDA) appeared in the top-3 for '
        '8/12 pathogens at fixed K=4. We then ran a specific grid of '
        'QDA × K=2..6 for each pathogen, and confirmed gains in 3 of them:'
    ))
    table_2col(doc, [
        ('Potato virus Y',        'GaussianNB K=4: 0.663 → QDA K=5: 0.675 (+0.012)'),
        ('Ralstonia solanacearum','GaussianNB K=4: 0.634 → QDA K=5: 0.681 (+0.047)'),
        ('Streptomyces scabies',  'GaussianNB K=3: 0.655 → QDA K=4: 0.676 (+0.021)'),
    ], header=['Pathogen', 'Before → After (QDA)'])
    para(doc, (
        'Why does QDA work for these 3? QDA estimates a covariance matrix per class, '
        'unlike GaussianNB which assumes full independence. '
        'For viruses and systemic bacteria, the k-mer frequency pattern '
        'of responsive lncRNAs seems to have inter-position correlations '
        'that QDA captures better. For pathogens with few positives '
        '(e.g. Meloidogyne, 47 samples), QDA overfits — GaussianNB is more robust.'
    ))

    # ── 10. CALIBRATION ──────────────────────────────────────────────────────
    heading(doc, '10. Probability calibration', level=1)
    para(doc, (
        'Before calibration, the web tool sometimes showed probabilities '
        'of 100% or 0% for many pathogens at once — which is biologically '
        'implausible. This happened because GaussianNB, when very confident, '
        'pushes probabilities to the extremes.'
    ))
    para(doc, (
        'Solution: CalibratedClassifierCV with sigmoid method (Platt scaling). '
        'This adds a logistic regression on top of the raw classifier outputs '
        'to map them to calibrated probabilities. '
        'Applied only to the final model — does not affect CV or F1.'
    ))
    code_block(doc, [
        '  from sklearn.calibration import CalibratedClassifierCV',
        '',
        '  raw_clf = GaussianNB()   # or QDA, KNN, etc.',
        '  clf = CalibratedClassifierCV(raw_clf, method="sigmoid", cv=max(2, n_cv))',
        '  clf.fit(X_all, y_all)    # train on full dataset',
        '',
        '  # Now predict_proba() returns calibrated probabilities',
        '  prob = clf.predict_proba(X_new)[0][1]  # prob. of positive class',
    ])
    para(doc, 'Result: instead of 0%/100%, probabilities became distributed (e.g. 33%, 51%, 28%), much more useful for prioritisation.')

    # ── 11. RESULTS ──────────────────────────────────────────────────────────
    heading(doc, '11. Final results per pathogen (pipeline v4)', level=1)
    para(doc, (
        'Complete metrics table for each model — all computed via '
        '5-fold stratified cross-validation. '
        'MCC = Matthews Correlation Coefficient (ranges from -1 to +1; '
        '0 = random prediction, 1 = perfect).'
    ))

    import json as _json
    from pathlib import Path as _Path
    cfg_path = _Path(__file__).parent.parent / 'models' / 'best_models_config.json'
    cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    pathogens_order = [
        'Alternaria solani', 'Globodera spp', 'Leptinotarsa decemlineata',
        'Meloidogyne javanica', 'Pectobacterium carotovorum', 'Phytophthora infestans',
        'Potato virus A', 'Potato virus Y', 'Ralstonia solanacearum',
        'Spongospora subterranea', 'Streptomyces scabies', 'Synchytrium endobioticum',
    ]
    model_map = {
        'Alternaria solani': ('K=3','GaussianNB'), 'Globodera spp': ('K=4','GaussianNB'),
        'Leptinotarsa decemlineata': ('K=4','GaussianNB'), 'Meloidogyne javanica': ('K=2','KNN'),
        'Pectobacterium carotovorum': ('K=4','Logistic Reg.'), 'Phytophthora infestans': ('K=2','SVC_RBF'),
        'Potato virus A': ('K=2','GaussianNB'), 'Potato virus Y': ('K=5','QDA'),
        'Ralstonia solanacearum': ('K=5','QDA'), 'Spongospora subterranea': ('K=4','GaussianNB'),
        'Streptomyces scabies': ('K=4','QDA'), 'Synchytrium endobioticum': ('K=3','GaussianNB'),
    }

    from docx.shared import Pt as _Pt
    from docx.enum.table import WD_TABLE_ALIGNMENT as _TA
    from docx.enum.text import WD_ALIGN_PARAGRAPH as _WA
    from docx.oxml.ns import qn as _qn
    from docx.oxml import OxmlElement as _OE
    from docx.shared import RGBColor as _RGB

    def _bg(cell, hex_c):
        tc = cell._tc; pr = tc.get_or_add_tcPr()
        s = _OE('w:shd'); s.set(_qn('w:val'),'clear')
        s.set(_qn('w:color'),'auto'); s.set(_qn('w:fill'), hex_c)
        pr.append(s)

    t = doc.add_table(rows=1, cols=8)
    t.style = 'Table Grid'
    t.alignment = _TA.CENTER
    for i, h in enumerate(['Pathogen', 'K / Model', 'F1', 'Precision', 'Recall', 'ROC-AUC', 'Accuracy', 'MCC']):
        c = t.rows[0].cells[i]
        c.text = ''
        run = c.paragraphs[0].add_run(h)
        run.bold = True; run.font.size = _Pt(9)
        c.paragraphs[0].alignment = _WA.CENTER
        _bg(c, '1565C0')
        run.font.color.rgb = _RGB(0xFF,0xFF,0xFF)

    for p_name in pathogens_order:
        m   = cfg.get(p_name, {})
        k_m = model_map.get(p_name, ('—','—'))
        row = t.add_row().cells
        for ci, val in enumerate([
            p_name, f'{k_m[0]}  {k_m[1]}',
            f'{m.get("f1",0):.4f}', f'{m.get("precision",0):.4f}',
            f'{m.get("recall",0):.4f}', f'{m.get("roc_auc",0):.4f}',
            f'{m.get("accuracy",0):.4f}', f'{m.get("mcc",0):.4f}',
        ]):
            row[ci].text = ''
            run = row[ci].paragraphs[0].add_run(val)
            run.font.size = _Pt(8.5)
            row[ci].paragraphs[0].alignment = _WA.LEFT if ci < 2 else _WA.CENTER
        if m.get('f1', 0) >= 0.70:
            _bg(row[2], 'C8E6C9')
        if m.get('roc_auc', 0) >= 0.70:
            _bg(row[5], 'C8E6C9')

    doc.add_paragraph()
    para(doc, 'What each metric means:', bold=True)
    bullet(doc, [
        ('F1-score: ',  'harmonic mean of precision and recall. Balances both — good for imbalanced classes.'),
        ('Precision: ', 'of the lncRNAs the model called "responsive", how many actually are? High precision = fewer false positives.'),
        ('Recall: ',    'of the lncRNAs that are truly responsive, how many did the model detect? High recall = fewer false negatives.'),
        ('ROC-AUC: ',   'area under the ROC curve. Measures class separability regardless of threshold. 0.5 = random, 1.0 = perfect.'),
        ('Accuracy: ',  'total proportion of correct predictions (positives + negatives).'),
        ('MCC: ',       'Matthews Correlation Coefficient. Robust metric considering all four cells of the confusion matrix. Ranges from -1 to +1.'),
    ])

    # ── 12. HOW TO RUN ───────────────────────────────────────────────────────
    heading(doc, '12. How to run the tool', level=1)
    para(doc, 'Requirements: Python 3.10+, install dependencies with:')
    code_block(doc, ['  pip install -r requirements.txt'])
    para(doc, 'Start the web application:')
    code_block(doc, [
        '  # Linux/Mac',
        '  ./run.sh',
        '',
        '  # Windows',
        '  run.bat',
        '',
        '  # Or directly:',
        '  python app.py',
    ])
    para(doc, 'Open http://localhost:5000 in your browser.')
    para(doc, (
        'If the sequence database (lncRNA.fa) is in a different location, '
        'set the BISPOLP_DATA_PATH environment variable:'
    ))
    code_block(doc, [
        '  BISPOLP_DATA_PATH=/my/path/to/potato_data python app.py',
    ])
    para(doc, (
        'Without the FASTA file, the tool works normally for predictions — '
        'only the "random sequence" button is disabled.'
    ))

    # ── 13. HOW TO RETRAIN ───────────────────────────────────────────────────
    heading(doc, '13. How to retrain the models', level=1)
    para(doc, (
        'The main training script is src/train.py. '
        'It uses the raw PotatoBSLnc data to retrain all 12 models from scratch.'
    ))
    code_block(doc, [
        '  # Run from BiSPoLP-ML/ directory',
        '  python src/train.py',
        '',
        '  # Models will be saved in models/saved/',
        '  # Grid search results in results/v4/',
        '  # Config JSON updated in models/best_models_config.json',
    ])
    para(doc, 'Estimated time: 30–60 minutes total, depending on hardware.')
    para(doc, 'The experiments that led to pipeline v4 are in experiments/:')
    table_2col(doc, [
        ('01_baseline_v1.py',    'Original pipeline (v1) with NEG_RATIO=3 + SMOTE — baseline reference.'),
        ('02_test_abordagens.py','Systematically tests 8 different approaches to try to improve F1.'),
        ('03_test_fwd_rev.py',   'Experiments with forward + reverse complement k-mers — no improvement.'),
        ('04_test_lazy.py',      'LazyPredict: screens ~30 classifiers + QDA × K=2..6 grid.'),
    ], header=['Script', 'What it does'])

    # ── 14. FILE STRUCTURE ───────────────────────────────────────────────────
    heading(doc, '14. Project file structure', level=1)
    code_block(doc, [
        'BiSPoLP-ML/',
        '├── app.py                  ← Flask web application (run this)',
        '├── run.sh / run.bat        ← Shortcuts to start',
        '├── requirements.txt        ← Python dependencies',
        '├── .gitignore',
        '│',
        '├── templates/',
        '│   └── index.html          ← Web interface (EN/PT-BR)',
        '│',
        '├── models/',
        '│   ├── best_models_config.json  ← Config: K, type, metrics per pathogen',
        '│   └── saved/              ← Trained models (.pkl)',
        '│       ├── Alternaria_solani_model.pkl',
        '│       ├── Alternaria_solani_selector.pkl',
        '│       ├── Alternaria_solani_pca.pkl',
        '│       ├── Alternaria_solani_scaler.pkl',
        '│       ├── Alternaria_solani_kmer_dict.pkl',
        '│       └── ... (5 files × 12 pathogens = 60 files)',
        '│',
        '├── src/',
        '│   ├── train.py            ← Full (re)training script',
        '│   └── predict.py          ← Alternative app version',
        '│',
        '├── experiments/',
        '│   ├── 01_baseline_v1.py',
        '│   ├── 02_test_abordagens.py',
        '│   ├── 03_test_fwd_rev.py',
        '│   ├── 04_test_lazy.py',
        '│   └── results/            ← Experiment CSVs',
        '│',
        '├── results/',
        '│   ├── v1/                 ← grid_search_results.csv, best_models.csv',
        '│   └── v4/                 ← grid_search_results.csv, best_models.csv',
        '│',
        '├── article/',
        '│   ├── generate_article.py',
        '│   ├── artigo_lncrna_potato_stress.docx  ← PT-BR article',
        '│   └── bispolp_article_en.docx            ← EN article',
        '│',
        '└── docs/',
        '    ├── gerar_documentacao.py',
        '    ├── COMO_FUNCIONA.docx    ← PT-BR documentation',
        '    └── HOW_IT_WORKS.docx     ← EN documentation',
    ])

    return doc


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    base = Path(__file__).parent

    print('Gerando COMO_FUNCIONA.docx (PT-BR)...')
    out_pt = base / 'COMO_FUNCIONA.docx'
    build().save(str(out_pt))
    print(f'  ✓ Salvo: {out_pt}  ({os.path.getsize(out_pt)//1024} KB)')

    print('Generating HOW_IT_WORKS.docx (EN)...')
    out_en = base / 'HOW_IT_WORKS.docx'
    build_en().save(str(out_en))
    print(f'  ✓ Saved: {out_en}  ({os.path.getsize(out_en)//1024} KB)')
