#  AMR Resistance Predictor — CodeCure DynamicDuo

<div align="center">

### CodeCure Biohackathon 2025 — Track B: Antibiotic Resistance Prediction

[ Live Tool](https://codecuredynamicduo-zsnpqygapp6lkcxnogxrmte.streamlit.app/) · [ Notebooks](./Notebooks/) · [ Outputs](./outputs/)

*Predicting antibiotic resistance in bacterial isolates by bridging phenotypic susceptibility data with CARD genomic resistance knowledge — enabling data-driven treatment guidance.*

</div>

---

## Problem Statement

AMR is a top WHO global health threat. This project addresses **Track B** of the CodeCure Biohackathon:

> *"Develop a model that predicts antibiotic resistance from bacterial genetic or phenotypic data, identify the most predictive features, and suggest treatment strategies."*

---

## Key Results

| Drug Class | Weighted F1 | Macro F1 | ROC-AUC | 5-Fold CV |
|:---:|:---:|:---:|:---:|:---:|
| Beta-lactam | 0.994 | 0.859 | 1.000 | 0.994 ± 0.002 |
| Aminoglycoside | 0.965 | 0.758 | 0.990 | 0.965 ± 0.011 |
| Quinolone | 0.959 | 0.752 | 0.990 | 0.970 ± 0.002 |
| Other | 0.976 | 0.722 | 0.996 | 0.987 ± 0.002 |

**Critical findings:** 60.6% of isolates resist a WHO Reserve antibiotic; 30.6% are MDR (≥3 classes). CTX-M/OXA-48 dominate beta-lactam resistance; QNR is the primary quinolone predictor — both biologically validated.

> Macro F1 is the meaningful headline metric. The high weighted F1 for beta-lactam reflects 94.7% resistance prevalence in the dataset, not model overconfidence.

---

## Pipeline

```
Kaggle (9,957) + Mendeley (274) + CARD ARO index
        ↓  merge + EUCAST v14 breakpoint conversion
   10,231 isolates × 70 columns
        ↓  phenotype-driven CARD enrichment + feature engineering
   Per-class curated feature sets (7–12 features, no leakage)
        ↓  RandomizedSearchCV → XGBoost (multi:softmax) × 4 + SMOTE
   SHAP analysis · Gene network · Streamlit decision tool
```

**Key innovation — phenotype-driven CARD enrichment:** Instead of mapping species → genes (zero variance), each isolate's *observed resistance phenotype* maps to CARD gene families, producing isolate-level variance with genuine biological signal.

---

## Repository Structure

```
CodeCure_DynamicDuo/
├── Notebooks/
│   ├── 01_Data_Integration_Preprocessing_FE.ipynb
│   └── 02_Model_Building_Feature_imp.ipynb
├── Data/
│   ├── raw/
│   └── processed/amr_final_v2.csv          ← 10,231 × 70
├── models/
│   ├── xgb_beta_lactam.pkl
│   ├── xgb_aminoglycoside.pkl
│   ├── xgb_quinolone.pkl
│   └── xgb_other.pkl
├── outputs/                                 ← confusion matrices, SHAP plots, gene network
├── app.py                                   ← Streamlit decision tool
├── gene_network.py
└── requirements.txt
```

---

## Methodology

### Data & Targets

| Source | Isolates | Notes |
|---|---|---|
| Kaggle multi-resistance | 9,957 | 15 antibiotics, pre-labelled R/I/S |
| Mendeley disk-diffusion | 274 | mm → R/I/S via EUCAST v14 |
| CARD ARO index v4.0.1 | — | Gene family → drug class enrichment |

Four drug-class targets via majority vote over constituent antibiotics (`2=R > 1=I > 0=S`, `NaN` if untested):
`target_beta_lactam` (6 drugs) · `target_aminoglycoside` (2) · `target_quinolone` (3) · `target_other` (4)

### Model Training

```python
XGBClassifier(objective='multi:softmax', num_class=3)
# Tuning: RandomizedSearchCV — 30 iterations, 3-fold CV, f1_weighted
# Imbalance: SMOTE + compute_sample_weight('balanced')
# Validation: 5-fold stratified CV + 20% held-out test
```

Per-class curated feature sets (7–12 features each) exclude all leaky columns derived from AST outcomes (`n_resistant_*`, `pct_resistant_*`, `is_mdr`, `phenotype_*`).

---

## SHAP Feature Importance

Top predictors per drug class (Resistant class, SHAP interventional):

| Rank | Beta-lactam | Aminoglycoside | Quinolone | Other |
|:---:|---|---|---|---|
| 1 | `card_has_oxa_48` | `card_relevant_gene_count` | `card_has_qnr` | `card_n_active_drug_classes` |
| 2 | `card_has_vim` | `card_gene_family_diversity` | `card_gene_family_diversity` | `card_gene_family_diversity` |
| 3 | `card_relevant_gene_count` | `age` | `card_mechanism_diversity` | `species_enc` |

All top features align with published AMR biology: OXA-48/VIM/KPC as dominant carbapenemases, QNR for plasmid-mediated quinolone resistance, and AME gene diversity driving aminoglycoside breadth.

---

## Decision-Support Tool

**Live:** [https://codecuredynamicduo-zsnpqygapp6lkcxnogxrmte.streamlit.app/](https://codecuredynamicduo-zsnpqygapp6lkcxnogxrmte.streamlit.app/)

Enter species, patient demographics, and detected CARD genes → get S/I/R predictions for all 4 drug classes with confidence bars, treatment guidance, and a **critical MDR alert** for WHO Reserve antibiotic resistance.

>  For research and educational use only. Clinical decisions require qualified healthcare professionals.

---

## Setup

```bash
git clone https://github.com/RokithArjunraj/CodeCure_DynamicDuo.git
cd CodeCure_DynamicDuo
pip install -r requirements.txt
```

Download raw data:

| Dataset | Source | Save to |
|---|---|---|
| Kaggle multi-resistance | [Kaggle](https://www.kaggle.com/datasets/adilimadeddinehosni/multi-resistance-antibiotic-susceptibility) | `Data/raw/` |
| Mendeley disk-diffusion | [Mendeley](https://data.mendeley.com/datasets/ccmrx8n7mk/1) | `Data/raw/Dataset.xlsx` |
| CARD aro_index.tsv | [CARD v4.0.1](https://card.mcmaster.ca/download) | `Data/card/aro_index.tsv` |

```bash
jupyter notebook Notebooks/01_Data_Integration_Preprocessing_FE.ipynb
jupyter notebook Notebooks/02_Model_Building_Feature_imp.ipynb
python gene_network.py
streamlit run app.py
```

**Requirements:** `pandas · numpy · scikit-learn · xgboost · imbalanced-learn · shap · matplotlib · seaborn · networkx · pyvis · streamlit · umap-learn · scipy · openpyxl`

---

## References

1. Alcock et al. 2023. CARD 2023. *Nucleic Acids Research*, **51**, D690–D699.
2. EUCAST. 2024. *Breakpoint Tables*, v14.0.
3. WHO. 2023. *AWaRe Antibiotic Book*.
4. Lundberg & Lee. 2017. SHAP. *NeurIPS 2017*.
5. Chen & Guestrin. 2016. XGBoost. *KDD 2016*.
6. Chawla et al. 2002. SMOTE. *JAIR*, **16**, 321–357.

---

<div align="center">

**DynamicDuo — CodeCure AI Biohackathon 2025 · Track B**

*XGBoost · SHAP · CARD · EUCAST v14 · Streamlit*

</div>
