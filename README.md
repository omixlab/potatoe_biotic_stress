# BiSPoLP — Biotic Stress Potato LncRNA Predictor

BiSPoLP is a machine learning tool that predicts which of 12 potato pathogens are likely to activate a given lncRNA during a biotic stress response, based on k-mer frequency composition of the input sequence.

The tool is available as a web application (Flask) with a bilingual interface (English / Portuguese).

**Author:** Christian Domingues Sanchez

---

## How it works

For each pathogen, BiSPoLP trains an independent binary classifier that learns to distinguish lncRNAs differentially expressed during infection (positive class) from a randomly sampled background of non-responsive lncRNAs (negative class). The input sequence is converted into a k-mer frequency vector, reduced via SelectKBest + PCA, scaled with StandardScaler, and classified. Probabilities are calibrated with Platt scaling (sigmoid) to avoid 0%/100% extremes.

**Pipeline (v4):**

```
Sequence → k-mer frequencies (K=2..6)
         → SelectKBest (100 features, ANOVA F)
         → PCA (50 components)
         → StandardScaler
         → Classifier (per-pathogen best)
         → CalibratedClassifierCV (sigmoid)
```

The best K and classifier for each pathogen were selected by 5-fold stratified cross-validation across 9 classifier families and K=2..6 — totaling 540 combinations evaluated.

---

## Classifiers evaluated

GaussianNB, BernoulliNB, QDA (Quadratic Discriminant Analysis), SVC_RBF, NuSVC_RBF, Logistic Regression, KNN (k=5), Random Forest (100 trees), MLP (100 hidden units)

Candidate classifiers were first screened with [LazyPredict](https://github.com/shankarpandala/lazypredict), which identified QDA as a top candidate alongside GaussianNB — both generative models that work well in high-dimensional spaces with limited positive samples.

---

## Results (5-fold CV, pipeline v4)

| Pathogen | Type | K | Model | F1 | ROC-AUC | MCC |
|---|---|:---:|---|:---:|:---:|:---:|
| Alternaria solani | Fungus | 3 | GaussianNB | 0.6735 | 0.7433 | 0.345 |
| Globodera spp | Nematode | 4 | GaussianNB | 0.6875 | 0.7141 | 0.304 |
| Leptinotarsa decemlineata | Insect | 4 | GaussianNB | 0.6543 | 0.6369 | 0.220 |
| Meloidogyne javanica | Nematode | 2 | KNN | 0.5879 | 0.5610 | 0.052 |
| Pectobacterium carotovorum | Bacterium | 4 | LogisticRegression | 0.6167 | 0.6838 | 0.239 |
| Phytophthora infestans | Oomycete | 2 | SVC_RBF | 0.6626 | 0.6758 | 0.262 |
| Potato virus A | Virus | 2 | GaussianNB | 0.6283 | 0.5881 | 0.136 |
| Potato virus Y | Virus | 5 | QDA | 0.6613 | 0.6270 | 0.200 |
| Ralstonia solanacearum | Bacterium | 5 | QDA | 0.6644 | 0.6495 | 0.237 |
| Spongospora subterranea | Protist | 4 | GaussianNB | 0.6519 | 0.6544 | 0.229 |
| Streptomyces scabies | Actinobacterium | 4 | QDA | 0.6619 | 0.6626 | 0.235 |
| Synchytrium endobioticum | Fungus | 3 | GaussianNB | 0.6518 | 0.6543 | 0.255 |
| **Mean** | | | | **0.6502** | **0.6542** | **0.243** |

MCC = Matthews Correlation Coefficient. All metrics are means over 5-fold stratified CV with preprocessing fitted inside each fold.

---

## Data

BiSPoLP was trained on the [PotatoBSLnc](https://bis.zju.edu.cn/PotatoBSLnc/) database — a public resource of lncRNAs differentially expressed in potato (*Solanum tuberosum*) under biotic stress.

- **Total sequences:** 18,636 lncRNAs
- **Positive class:** lncRNAs with |log₂FC| > 1 in differential expression analysis per pathogen
- **Negative class:** randomly sampled background (1:1 ratio with positives per pathogen)

The raw data is **not included** in this repository. Set the environment variable `BISPOLP_DATA_PATH` to point to your local copy of the PotatoBSLnc dataset before running the application or retraining.

---

## Setup

**Requirements:** [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate bispolp

# 2. Run the application
python app.py
```

Then open **http://localhost:5000** in your browser.

Alternatively, use the run script:

```bash
./run.sh

# With a custom data path:
BISPOLP_DATA_PATH=/path/to/potato_data ./run.sh
```

---

## Retraining

To retrain all 12 models from scratch:

```bash
# Set the path to the PotatoBSLnc dataset
export BISPOLP_DATA_PATH=/path/to/potato_data

python src/train.py
```

Models are saved to `models/saved/` and grid search results to `results/v4/`.

To recompute all evaluation metrics (F1, precision, recall, ROC-AUC, accuracy, MCC):

```bash
python src/predict.py   # or rerun compute_metrics.py from the project root
```

---

## Project structure

```
BiSPoLP-ML/
├── app.py                  # Flask web application (entry point)
├── environment.yml         # Conda environment definition
├── requirements.txt        # pip requirements (alternative to conda)
├── run.sh                  # Convenience startup script
│
├── templates/
│   └── index.html          # Web interface (EN / PT-BR)
│
├── models/
│   ├── saved/              # Trained .pkl files (60 files: 5 per pathogen)
│   └── best_models_config.json  # Best K, classifier and all CV metrics
│
├── src/
│   ├── train.py            # Full training pipeline (grid search, save models)
│   └── predict.py          # Prediction utilities
│
├── experiments/            # Exploratory scripts used during development
│   ├── 01_baseline_v1.py   # Baseline pipeline v1
│   ├── 02_test_abordagens.py
│   ├── 03_test_fwd_rev.py
│   └── 04_test_lazy.py     # LazyPredict screening
│
├── results/
│   ├── v1/                 # Baseline results (SMOTE, no PCA)
│   └── v4/                 # Final pipeline results
│
├── article/
│   └── generate_article.py   # Article generator (python-docx)
│                              # .docx outputs not versioned (pending publication)
│
└── docs/
    └── gerar_documentacao.py  # Documentation generator (PT-BR + EN)
                               # .docx outputs not versioned (pending publication)
```

---

## Article and documentation

The scientific article and technical documentation are generated locally via:

```bash
python article/generate_article.py   # generates PT-BR and EN .docx
python docs/gerar_documentacao.py    # generates COMO_FUNCIONA.docx and HOW_IT_WORKS.docx
```

These files are not versioned in this repository as the article is currently under preparation for submission.

---

## Citation

If you use BiSPoLP in your research, please cite:

> Sanchez, C.D. (2025). *BiSPoLP: Biotic Stress Potato LncRNA Predictor — a machine learning tool for lncRNA classification under biotic stress in Solanum tuberosum.* [Manuscript in preparation]

---

## License

This project is intended for academic and research use.
