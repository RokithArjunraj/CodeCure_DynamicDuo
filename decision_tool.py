"""
AMR Decision-Support Tool
CodeCure Biohackathon — Track B
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AMR Resistance Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #e8f4f8; font-family: 'Courier New', monospace; }
    h2, h3 { color: #a8d8ea; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .resistant   { border-left: 4px solid #e74c3c; }
    .susceptible { border-left: 4px solid #2ecc71; }
    .intermediate{ border-left: 4px solid #f39c12; }
    .critical-alert {
        background: #3d0000;
        border: 2px solid #e74c3c;
        border-radius: 8px;
        padding: 16px;
        color: #ff6b6b;
        font-weight: bold;
        margin: 8px 0;
    }
    .safe-alert {
        background: #003d1a;
        border: 2px solid #2ecc71;
        border-radius: 8px;
        padding: 16px;
        color: #55efc4;
        margin: 8px 0;
    }
    .info-box {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 6px;
        padding: 12px;
        color: #a8d8ea;
        font-size: 0.85em;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load trained XGBoost models. Falls back to demo mode if not found."""
    models = {}
    model_files = {
        'target_beta_lactam'   : 'models/xgb_beta_lactam.pkl',
        'target_aminoglycoside': 'models/xgb_aminoglycoside.pkl',
        'target_quinolone'     : 'models/xgb_quinolone.pkl',
        'target_other'         : 'models/xgb_other.pkl',
    }
    for target, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[target] = pickle.load(f)
    return models


# ── Species encoder mapping (must match training) ─────────────
SPECIES_ENC = {
    'escherichia coli'      : 0,
    'klebsiella pneumoniae' : 1,
    'proteus mirabilis'     : 2,
    'citrobacter freundii'  : 3,
    'morganella morganii'   : 4,
    'serratia marcescens'   : 5,
    'pseudomonas aeruginosa': 6,
    'acinetobacter baumannii':7,
    'enterobacter cloacae'  : 8,
    'enterococcus faecalis' : 9,
}

# ── Clean feature order — must match training exactly ─────────
CLEAN_FEATURES = [
    'species_enc', 'age', 'gender_enc', 'has_diabetes',
    'has_hypertension', 'prior_hospitalisation', 'is_mendeley',
    'card_n_active_drug_classes', 'card_relevant_gene_count',
    'card_mechanism_diversity', 'card_gene_family_diversity',
    'card_has_target_protection', 'card_has_reduced_permeability',
    'card_has_ctx_m', 'card_has_tem', 'card_has_kpc',
    'card_has_ndm', 'card_has_vim', 'card_has_imp',
    'card_has_oxa_48', 'card_has_mcr', 'card_has_qnr',
    'card_has_aac', 'card_has_rnd_efflux', 'card_has_mfs_efflux',
]

TARGET_LABELS = {
    'target_beta_lactam'   : 'Beta-lactam',
    'target_aminoglycoside': 'Aminoglycoside',
    'target_quinolone'     : 'Quinolone',
    'target_other'         : 'Other antibiotics',
}

CLASS_LABEL = {0: 'Susceptible', 1: 'Intermediate', 2: 'Resistant'}
CLASS_COLOR = {0: '#2ecc71',     1: '#f39c12',       2: '#e74c3c'}
CLASS_EMOJI = {0: '✅',           1: '⚠️',             2: '❌'}

# WHO Reserve antibiotics per class
RESERVE_DRUGS = {
    'target_beta_lactam'   : 'Imipenem (carbapenem)',
    'target_other'         : 'Colistin (polymyxin)',
}

# Treatment suggestions per drug class and resistance level
TREATMENT_GUIDE = {
    'target_beta_lactam': {
        0: "Beta-lactams (amoxicillin, cephalosporins) likely effective.",
        1: "Use higher-dose beta-lactam or add beta-lactamase inhibitor (e.g. piperacillin-tazobactam).",
        2: "Avoid beta-lactams. Consider carbapenem-sparing regimen. Check for ESBL/carbapenemase.",
    },
    'target_aminoglycoside': {
        0: "Gentamicin or amikacin can be used as adjunct therapy.",
        1: "Use amikacin (if gentamicin-intermediate). Monitor renal function.",
        2: "Avoid aminoglycosides. Consider alternative gram-negative coverage.",
    },
    'target_quinolone': {
        0: "Ciprofloxacin or levofloxacin appropriate for susceptible isolates.",
        1: "Quinolone use not recommended — risk of treatment failure.",
        2: "Avoid all fluoroquinolones. QNR gene likely present — consider alternatives.",
    },
    'target_other': {
        0: "Cotrimoxazole, nitrofurantoin, or chloramphenicol may be options depending on infection site.",
        1: "Use with caution. Confirm with sensitivity testing.",
        2: "Colistin resistance is clinically critical. Combination therapy required — consult infectious disease specialist.",
    },
}

# ─────────────────────────────────────────────────────────────
# DEMO PREDICTION (when models not loaded)
# Uses simplified heuristic rules based on CARD features
# ─────────────────────────────────────────────────────────────
def demo_predict(features_dict):
    """Heuristic fallback prediction for demo mode."""
    preds, probs = {}, {}

    ctx_m = features_dict.get('card_has_ctx_m', 0)
    kpc   = features_dict.get('card_has_kpc', 0)
    ndm   = features_dict.get('card_has_ndm', 0)
    vim   = features_dict.get('card_has_vim', 0)
    imp_g = features_dict.get('card_has_imp', 0)
    mcr   = features_dict.get('card_has_mcr', 0)
    qnr   = features_dict.get('card_has_qnr', 0)
    aac   = features_dict.get('card_has_aac', 0)
    n_act = features_dict.get('card_n_active_drug_classes', 0)
    sp    = features_dict.get('species_enc', 0)

    # Beta-lactam
    if kpc or ndm or vim or imp_g:
        preds['target_beta_lactam'] = 2
        probs['target_beta_lactam'] = [0.02, 0.03, 0.95]
    elif ctx_m:
        preds['target_beta_lactam'] = 2
        probs['target_beta_lactam'] = [0.05, 0.10, 0.85]
    else:
        preds['target_beta_lactam'] = 0
        probs['target_beta_lactam'] = [0.80, 0.12, 0.08]

    # Aminoglycoside
    if aac and n_act >= 3:
        preds['target_aminoglycoside'] = 2
        probs['target_aminoglycoside'] = [0.08, 0.12, 0.80]
    elif aac:
        preds['target_aminoglycoside'] = 1
        probs['target_aminoglycoside'] = [0.30, 0.45, 0.25]
    else:
        preds['target_aminoglycoside'] = 0
        probs['target_aminoglycoside'] = [0.75, 0.15, 0.10]

    # Quinolone
    if qnr and n_act >= 4:
        preds['target_quinolone'] = 2
        probs['target_quinolone'] = [0.05, 0.10, 0.85]
    elif qnr:
        preds['target_quinolone'] = 1
        probs['target_quinolone'] = [0.25, 0.45, 0.30]
    else:
        preds['target_quinolone'] = 0
        probs['target_quinolone'] = [0.70, 0.20, 0.10]

    # Other
    if mcr:
        preds['target_other'] = 2
        probs['target_other'] = [0.05, 0.10, 0.85]
    elif n_act >= 5:
        preds['target_other'] = 2
        probs['target_other'] = [0.15, 0.20, 0.65]
    else:
        preds['target_other'] = 0
        probs['target_other'] = [0.70, 0.18, 0.12]

    return preds, probs


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
models = load_models()
demo_mode = len(models) == 0

# ── Header ────────────────────────────────────────────────────
st.title("🦠 AMR Resistance Predictor")
st.markdown(
    "**CodeCure Biohackathon — Track B** | "
    "Predicts antibiotic resistance class for bacterial isolates "
    "using XGBoost + CARD gene database enrichment"
)

if demo_mode:
    st.info(
        "ℹ️ **Demo mode** — trained model files not found in `models/`. "
        "Running heuristic predictions based on CARD gene logic. "
        "Save your trained models with `pickle.dump(model, open('models/xgb_beta_lactam.pkl','wb'))` "
        "to enable full ML predictions.",
        icon="ℹ️"
    )

st.divider()

# ── Sidebar — inputs ──────────────────────────────────────────
with st.sidebar:
    st.header("🔬 Isolate Details")

    st.subheader("Patient & Isolate")
    species = st.selectbox(
        "Bacterial species",
        options=list(SPECIES_ENC.keys()),
        format_func=lambda x: x.title(),
    )
    age = st.slider("Patient age", 1, 95, 45)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

    st.subheader("Clinical Risk Factors")
    diabetes      = st.checkbox("Diabetes")
    hypertension  = st.checkbox("Hypertension")
    prior_hosp    = st.checkbox("Prior hospitalisation")

    st.subheader("🧬 CARD Resistance Genes")
    st.caption("Select genes detected in this isolate")

    col1, col2 = st.columns(2)
    with col1:
        ctx_m  = st.checkbox("CTX-M (ESBL)", help="Extended-spectrum beta-lactamase — cephalosporin resistance")
        tem    = st.checkbox("TEM",           help="TEM beta-lactamase — penicillin resistance")
        kpc    = st.checkbox("KPC",           help="Klebsiella pneumoniae carbapenemase — Reserve antibiotic resistance")
        ndm    = st.checkbox("NDM",           help="New Delhi metallo-beta-lactamase — broad carbapenem resistance")
        vim    = st.checkbox("VIM",           help="Verona integron-encoded MBL — carbapenem resistance")
    with col2:
        imp_g  = st.checkbox("IMP",           help="Imipenem-resistant MBL")
        oxa_48 = st.checkbox("OXA-48",        help="OXA-48 carbapenemase")
        mcr    = st.checkbox("MCR",           help="Colistin resistance gene — last-resort antibiotic")
        qnr    = st.checkbox("QNR",           help="Quinolone resistance protein")
        aac    = st.checkbox("AAC",           help="Aminoglycoside-modifying enzyme")

    st.subheader("Mechanism Flags")
    rnd_efflux = st.checkbox("RND efflux pump",  help="Resistance-nodulation-cell division efflux")
    mfs_efflux = st.checkbox("MFS efflux pump",  help="Major facilitator superfamily efflux")
    tgt_prot   = st.checkbox("Target protection",help="Resistance via target protection mechanism")
    red_perm   = st.checkbox("Reduced permeability", help="Porin loss — reduced drug entry")

    predict_btn = st.button("🔍 Predict Resistance", use_container_width=True, type="primary")


# ── Build feature vector ──────────────────────────────────────
n_active = sum([ctx_m, tem, kpc, ndm, vim, imp_g, oxa_48, mcr, qnr, aac,
                rnd_efflux, mfs_efflux])

# Estimate continuous CARD features from binary flags
gene_count  = n_active * 8   # rough gene count per family
mech_div    = sum([
    int(any([ctx_m, tem, kpc, ndm, vim, imp_g, oxa_48])),  # inactivation
    int(rnd_efflux or mfs_efflux),                          # efflux
    int(tgt_prot),                                          # protection
    int(red_perm),                                          # permeability
    int(qnr),                                               # target alteration
])
gene_fam_div = n_active
total_antibiotics_tested = n_active

features_dict = {
    'species_enc'               : SPECIES_ENC.get(species, 0),
    'age'                       : age,
    'gender_enc'                : 1 if gender == 'Female' else 0,
    'has_diabetes'              : int(diabetes),
    'has_hypertension'          : int(hypertension),
    'prior_hospitalisation'     : int(prior_hosp),
    'is_mendeley'               : 0,
    'total_antibiotics_tested': total_antibiotics_tested,
    'card_n_active_drug_classes': n_active,
    'card_relevant_gene_count'  : gene_count,
    'card_mechanism_diversity'  : mech_div,
    'card_gene_family_diversity': gene_fam_div,
    'card_has_target_protection': int(tgt_prot),
    'card_has_reduced_permeability': int(red_perm),
    'card_has_ctx_m'            : int(ctx_m),
    'card_has_tem'              : int(tem),
    'card_has_kpc'              : int(kpc),
    'card_has_ndm'              : int(ndm),
    'card_has_vim'              : int(vim),
    'card_has_imp'              : int(imp_g),
    'card_has_oxa_48'           : int(oxa_48),
    'card_has_mcr'              : int(mcr),
    'card_has_qnr'              : int(qnr),
    'card_has_aac'              : int(aac),
    'card_has_rnd_efflux'       : int(rnd_efflux),
    'card_has_mfs_efflux'       : int(mfs_efflux),
}

#X_input = pd.DataFrame([features_dict])[CLEAN_FEATURES]
X_full = pd.DataFrame([features_dict])[CLEAN_FEATURES]

FEATURE_SETS = {
    'target_beta_lactam': [
        'species_enc','age','card_relevant_gene_count','card_has_oxa_48',
        'card_has_vim','card_gene_family_diversity','card_has_kpc',
        'card_has_tem','card_has_imp'
    ],
    'target_aminoglycoside': [
        'age','card_relevant_gene_count','card_gene_family_diversity',
        'species_enc','card_n_active_drug_classes','card_mechanism_diversity',
        'prior_hospitalisation','has_hypertension','has_diabetes',
        'gender_enc','total_antibiotics_tested'
    ],
    'target_quinolone': [
        'card_has_qnr','age','card_has_target_protection',
        'card_mechanism_diversity','card_relevant_gene_count',
        'species_enc','card_gene_family_diversity'
    ],
    'target_other': [
        'age','card_n_active_drug_classes','species_enc',
        'card_relevant_gene_count','card_gene_family_diversity'
    ]
}

# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    if models:
        predictions, probabilities = {}, {}
        for target, model in models.items():
            feature_list = FEATURE_SETS[target]
            #X_input = X_full[feature_list]
            X_input = X_full.reindex(columns=feature_list, fill_value=0)
            pred = int(model.predict(X_input)[0])
            prob = model.predict_proba(X_input)[0].tolist()
            predictions[target] = pred
            probabilities[target] = prob
    else:
        predictions, probabilities = demo_predict(features_dict)

    st.header(f"Results for {species.title()}")

    # ── MDR assessment ────────────────────────────────────────
    n_resistant  = sum(1 for v in predictions.values() if v == 2)
    is_mdr       = n_resistant >= 3
    is_critical  = (predictions.get('target_beta_lactam') == 2 and
                    any([kpc, ndm, vim, imp_g, oxa_48]))
    is_colistin_r = predictions.get('target_other') == 2 and mcr

    if is_critical or is_colistin_r:
        st.markdown(
            '<div class="critical-alert">⚠️ CRITICAL MDR — Resistant to WHO Reserve antibiotic. '
            'Immediate infectious disease consultation required.</div>',
            unsafe_allow_html=True
        )
    elif is_mdr:
        st.warning(f"⚠️ Multi-Drug Resistant (MDR) — Resistant to {n_resistant}/4 antibiotic classes")
    else:
        st.markdown(
            f'<div class="safe-alert">✅ Not MDR — Resistant to {n_resistant}/4 antibiotic classes</div>',
            unsafe_allow_html=True
        )

    # ── 4-class result cards ──────────────────────────────────
    st.subheader("Resistance Predictions by Drug Class")
    cols = st.columns(4)

    for i, (target, label) in enumerate(TARGET_LABELS.items()):
        pred  = predictions.get(target, 0)
        prob  = probabilities.get(target, [0.33, 0.33, 0.34])
        clr   = CLASS_COLOR[pred]
        emoji = CLASS_EMOJI[pred]
        cls   = CLASS_LABEL[pred]

        with cols[i]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:1.8em">{emoji}</div>'
                f'<div style="color:{clr};font-weight:bold;font-size:1.1em">{cls}</div>'
                f'<div style="color:#888;font-size:0.85em">{label}</div>'
                f'<div style="color:#ccc;font-size:0.8em;margin-top:8px">'
                f'Confidence: {max(prob)*100:.0f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Probability bars ──────────────────────────────────────
    st.subheader("Prediction Confidence")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    fig.patch.set_facecolor('#0f1117')

    for i, (target, label) in enumerate(TARGET_LABELS.items()):
        prob = probabilities.get(target, [0.33, 0.33, 0.34])
        ax   = axes[i]
        ax.set_facecolor('#1e2130')
        bars = ax.bar(['S', 'I', 'R'], prob,
                      color=['#2ecc71','#f39c12','#e74c3c'],
                      edgecolor='none', width=0.5)
        for bar, p in zip(bars, prob):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02, f'{p:.2f}',
                    ha='center', color='white', fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(label, color='#a8d8ea', fontsize=10, pad=6)
        ax.tick_params(colors='#888')
        ax.spines[:].set_visible(False)
        ax.yaxis.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Treatment guidance ────────────────────────────────────
    st.subheader("🩺 Treatment Guidance")
    for target, label in TARGET_LABELS.items():
        pred = predictions.get(target, 0)
        cls  = CLASS_LABEL[pred]
        clr  = CLASS_COLOR[pred]
        tip  = TREATMENT_GUIDE[target][pred]
        reserve_note = ""
        if target in RESERVE_DRUGS and pred == 2:
            reserve_note = f" ⚠️ **{RESERVE_DRUGS[target]} — WHO Reserve antibiotic**"

        with st.expander(f"{CLASS_EMOJI[pred]} **{label}** — {cls}{reserve_note}"):
            st.markdown(f'<div class="info-box">{tip}</div>', unsafe_allow_html=True)
            s_pct = f"{probabilities[target][0]*100:.0f}%"
            i_pct = f"{probabilities[target][1]*100:.0f}%"
            r_pct = f"{probabilities[target][2]*100:.0f}%"
            st.caption(f"Probabilities — S: {s_pct}  I: {i_pct}  R: {r_pct}")

    # ── Gene profile summary ──────────────────────────────────
    st.subheader("🧬 Gene Profile Summary")
    genes_present = {
        'CTX-M (ESBL)': ctx_m, 'TEM': tem, 'KPC': kpc, 'NDM': ndm,
        'VIM': vim, 'IMP': imp_g, 'OXA-48': oxa_48, 'MCR': mcr,
        'QNR': qnr, 'AAC': aac, 'RND efflux': rnd_efflux, 'MFS efflux': mfs_efflux,
    }
    detected = [g for g, v in genes_present.items() if v]
    if detected:
        gene_cols = st.columns(min(len(detected), 4))
        for j, gene in enumerate(detected):
            gene_cols[j % 4].markdown(
                f'<div class="metric-card" style="border-left:4px solid #e74c3c">'
                f'<div style="color:#e74c3c;font-size:0.9em;font-weight:bold">{gene}</div>'
                f'<div style="color:#888;font-size:0.75em">Detected</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No resistance genes selected. Predictions based on species and demographics only.")

else:
    # ── Landing state ─────────────────────────────────────────
    st.info("👈 Fill in isolate details in the sidebar and click **Predict Resistance**")

    st.subheader("📊 Model Performance Summary")
    perf_data = {
        'Drug Class'   : ['Beta-lactam', 'Aminoglycoside', 'Quinolone', 'Other'],
        'Weighted F1'  : [0.995, 0.961, 0.967, 0.980],
        'Macro F1'     : [0.881, 0.742, 0.763, 0.761],
        'ROC-AUC'      : [1.000, 0.990, 0.992, 0.997],
        'CV Mean ± Std': ['0.994 ± 0.002','0.966 ± 0.011','0.973 ± 0.006','0.988 ± 0.003'],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.subheader("🔬 Pipeline Overview")
    st.markdown("""
    | Stage | Details |
    |-------|---------|
    | **Data sources** | Kaggle multi-resistance (9,957 isolates) + Mendeley disk-diffusion (274) |
    | **CARD enrichment** | Phenotype-driven — isolate resistance → CARD gene lookup |
    | **Features** | 25 clean features: species, demographics, 13 CARD gene flags |
    | **Models** | XGBoost (400 trees) per drug class, SMOTE for class imbalance |
    | **Targets** | 4 drug-class resistance labels: Beta-lactam, Aminoglycoside, Quinolone, Other |
    | **Validation** | 5-fold stratified CV + held-out 20% test set |
    """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚕️ This tool is for research and educational purposes only. "
    "Clinical treatment decisions must be made by qualified healthcare professionals. "
    "| CodeCure Biohackathon 2025 — Track B | XGBoost + CARD AMR Database"
)
