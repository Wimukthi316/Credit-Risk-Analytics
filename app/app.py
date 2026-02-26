import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Global ── */
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

        /* ── Header banner ── */
        .header-banner {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
        }
        .header-banner h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
        .header-banner p  { font-size: 1rem; opacity: .75; margin: .4rem 0 0; }

        /* ── Section labels ── */
        .section-label {
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            color: #6b7280;
            margin-bottom: .4rem;
        }

        /* ── Cards ── */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,.06);
        }
        .metric-card .label { font-size: .8rem; color: #6b7280; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
        .metric-card .value { font-size: 2rem; font-weight: 700; margin-top: .25rem; }

        /* ── Decision banners ── */
        .decision-approve {
            background: #ecfdf5; border-left: 6px solid #10b981;
            padding: 1.2rem 1.5rem; border-radius: 10px; color: #065f46;
        }
        .decision-review  {
            background: #fffbeb; border-left: 6px solid #f59e0b;
            padding: 1.2rem 1.5rem; border-radius: 10px; color: #78350f;
        }
        .decision-reject  {
            background: #fef2f2; border-left: 6px solid #ef4444;
            padding: 1.2rem 1.5rem; border-radius: 10px; color: #7f1d1d;
        }
        .decision-title { font-size: 1.3rem; font-weight: 700; margin-bottom: .3rem; }
        .decision-body  { font-size: .95rem; line-height: 1.5; }

        /* ── Progress bar wrapper ── */
        .risk-bar-wrap { background: #f3f4f6; border-radius: 9999px; height: 14px; overflow: hidden; margin: .5rem 0 .2rem; }
        .risk-bar      { height: 14px; border-radius: 9999px; transition: width .6s ease; }

        /* ── Predict button ── */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #0f3460, #533483);
            color: white; font-weight: 700; font-size: 1.05rem;
            padding: .75rem 2rem; border-radius: 10px; border: none;
            width: 100%; letter-spacing: .04em;
            box-shadow: 0 4px 14px rgba(15,52,96,.35);
            transition: opacity .2s;
        }
        div[data-testid="stButton"] > button:hover { opacity: .88; }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem; }

        /* ── Divider ── */
        hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(BASE_DIR, "..")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "credit_risk_model.pkl")
DATA_PATH  = os.path.join(ROOT_DIR, "data", "raw", "credit_risk_dataset.csv")

# ─────────────────────────────────────────────
#  Train & save model (runs only if pkl missing)
# ─────────────────────────────────────────────
def train_and_save_model():
    """Reproduce the exact training pipeline from the notebook and pickle the result."""
    df = pd.read_csv(DATA_PATH)

    # --- Cleaning ---
    df = df[df['person_age'] <= 100]
    df = df[(df['person_emp_length'] <= 100) | (df['person_emp_length'].isna())]
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate']     = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # --- Encoding ---
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)

    return rf

# ─────────────────────────────────────────────
#  Load model (train on-the-fly if pkl missing)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up model — this takes ~30 seconds on first run…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="header-banner">
        <h1>🏦 Credit Risk Prediction Dashboard</h1>
        <p>AI-powered loan default risk assessment · Random Forest · Real-time scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Sidebar – input form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("Fill in the loan application information below.")
    st.markdown("---")

    # ── Personal Information ──────────────────
    st.markdown('<p class="section-label">👤 Personal Information</p>', unsafe_allow_html=True)

    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input(
        "Annual Income ($)", min_value=1000, max_value=10_000_000,
        value=50_000, step=1000, format="%d",
    )
    emp_length = st.number_input(
        "Employment Length (years)", min_value=0.0, max_value=60.0, value=3.0, step=0.5,
    )
    home_ownership = st.selectbox(
        "Home Ownership",
        options=["MORTGAGE", "OWN", "RENT", "OTHER"],
    )

    st.markdown("---")

    # ── Loan Details ─────────────────────────
    st.markdown('<p class="section-label">💳 Loan Details</p>', unsafe_allow_html=True)

    loan_amount = st.number_input(
        "Loan Amount ($)", min_value=500, max_value=500_000,
        value=10_000, step=500, format="%d",
    )
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.0, step=0.1,
    )
    loan_intent = st.selectbox(
        "Loan Intent",
        options=["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
    )
    loan_grade = st.selectbox(
        "Loan Grade",
        options=["A", "B", "C", "D", "E", "F", "G"],
        help="A = Best credit profile · G = Highest risk",
    )

    st.markdown("---")

    # ── Credit History ────────────────────────
    st.markdown('<p class="section-label">📊 Credit History</p>', unsafe_allow_html=True)

    cred_hist_length = st.number_input(
        "Credit History Length (years)", min_value=0, max_value=50, value=5, step=1,
    )
    historical_default = st.radio(
        "Previous Default on File?",
        options=["No", "Yes"],
        horizontal=True,
    )

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

# ─────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

# Exact feature order the model was trained on
FEATURE_COLUMNS = [
    "person_age", "person_income", "person_emp_length",
    "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file",
    "cb_person_cred_hist_length",
    # home ownership dummies (MORTGAGE dropped)
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    # loan intent dummies (DEBTCONSOLIDATION dropped)
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
]


def preprocess(age, income, emp_length, loan_amount, interest_rate,
               loan_intent, loan_grade, cred_hist_length,
               home_ownership, historical_default):
    """Build the feature vector expected by the trained Random Forest model."""

    # 1. Numeric / ordinal features
    grade_encoded   = GRADE_MAP[loan_grade]
    default_encoded = 1 if historical_default == "Yes" else 0
    loan_pct_income = round(loan_amount / income, 4) if income > 0 else 0.0

    # 2. One-hot: person_home_ownership  (drop_first=True drops MORTGAGE)
    home_choices = ["OTHER", "OWN", "RENT"]
    home_ohe = {f"person_home_ownership_{c}": int(home_ownership == c) for c in home_choices}

    # 3. One-hot: loan_intent  (drop_first=True drops DEBTCONSOLIDATION)
    intent_choices = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    intent_ohe = {f"loan_intent_{c}": int(loan_intent == c) for c in intent_choices}

    # 4. Assemble in training order
    row = {
        "person_age":              age,
        "person_income":           income,
        "person_emp_length":       emp_length,
        "loan_grade":              grade_encoded,
        "loan_amnt":               loan_amount,
        "loan_int_rate":           interest_rate,
        "loan_percent_income":     loan_pct_income,
        "cb_person_default_on_file": default_encoded,
        "cb_person_cred_hist_length": cred_hist_length,
        **home_ohe,
        **intent_ohe,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


# ─────────────────────────────────────────────
#  Main panel – default state
# ─────────────────────────────────────────────
if not predict_clicked:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """<div class="metric-card">
                <div class="label">Model</div>
                <div class="value" style="font-size:1.2rem;color:#0f3460;">Random Forest</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """<div class="metric-card">
                <div class="label">Features Used</div>
                <div class="value" style="color:#0f3460;">17</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """<div class="metric-card">
                <div class="label">Decision Bands</div>
                <div class="value" style="font-size:1.1rem;color:#0f3460;">Approve / Review / Reject</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Fill in the applicant details in the sidebar and click **Predict Risk** to get started.")

    # Explanation table
    with st.expander("ℹ️  How does the scoring work?", expanded=False):
        st.markdown(
            """
            | Risk Probability | Decision | Meaning |
            |---|---|---|
            | **0 – 30 %** | ✅ **APPROVE** | Low default risk — loan recommended |
            | **30 – 60 %** | ⚠️ **REVIEW** | Moderate risk — manual underwriter check advised |
            | **> 60 %** | ❌ **REJECT** | High default risk — loan not recommended |

            **Expected Financial Loss** = Probability of Default × Loan Amount

            The model was trained on a Random Forest classifier with one-hot encoding for
            categorical variables and ordinal encoding for loan grade (A=0 … G=6).
            """
        )

# ─────────────────────────────────────────────
#  Prediction output
# ─────────────────────────────────────────────
if predict_clicked:
    input_df = preprocess(
        age, income, emp_length, loan_amount, interest_rate,
        loan_intent, loan_grade, cred_hist_length,
        home_ownership, historical_default,
    )

    proba        = model.predict_proba(input_df)[0][1]   # P(default)
    expected_loss = proba * loan_amount
    loan_pct_income = loan_amount / income * 100         # for display

    # ── Decision logic ──────────────────────
    if proba > 0.60:
        decision = "REJECT"
        decision_class = "decision-reject"
        decision_icon  = "❌"
        decision_msg   = (
            f"High probability of default ({proba*100:.1f}%). "
            "This application does not meet minimum credit standards. "
            "Disbursing this loan is not recommended."
        )
    elif proba > 0.30:
        decision = "REVIEW"
        decision_class = "decision-review"
        decision_icon  = "⚠️"
        decision_msg   = (
            f"Moderate default risk ({proba*100:.1f}%). "
            "This application requires manual underwriter review before a final decision. "
            "Consider requesting additional documentation or a co-signer."
        )
    else:
        decision = "APPROVE"
        decision_class = "decision-approve"
        decision_icon  = "✅"
        decision_msg   = (
            f"Low probability of default ({proba*100:.1f}%). "
            "The applicant meets the credit criteria. "
            "This loan can be approved under standard terms."
        )

    # ── Bar colour ────────────────────────────
    if proba > 0.60:
        bar_colour = "#ef4444"
    elif proba > 0.30:
        bar_colour = "#f59e0b"
    else:
        bar_colour = "#10b981"

    bar_pct = f"{proba * 100:.1f}%"

    # ── Layout ───────────────────────────────
    # Row 1 – KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Probability of Default</div>
                <div class="value" style="color:{bar_colour};">{proba*100:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Expected Financial Loss</div>
                <div class="value" style="color:#1e40af;">${expected_loss:,.0f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Loan Amount</div>
                <div class="value" style="color:#374151;">${loan_amount:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Loan as % of Income</div>
                <div class="value" style="color:#374151;">{loan_pct_income:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Row 2 – Risk bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Risk Probability Meter")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:.85rem;color:#6b7280;width:3rem;">0%</span>
            <div class="risk-bar-wrap" style="flex:1;">
                <div class="risk-bar" style="width:{bar_pct}; background:{bar_colour};"></div>
            </div>
            <span style="font-size:.85rem;color:#6b7280;width:3rem; text-align:right;">100%</span>
            <span style="font-size:1.1rem; font-weight:700; color:{bar_colour}; min-width:4rem; text-align:right;">{bar_pct}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding: 0 3.5rem 0 3.5rem; font-size:.7rem; color:#9ca3af;">
            <span>▲ APPROVE (&lt;30%)</span>
            <span>▲ REVIEW (30-60%)</span>
            <span>▲ REJECT (&gt;60%)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 3 – Decision banner
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="{decision_class}">
            <div class="decision-title">{decision_icon} {decision}</div>
            <div class="decision-body">{decision_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 4 – Application summary
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 Full Application Summary", expanded=True):
        left, right = st.columns(2)

        with left:
            st.markdown("**Personal Details**")
            st.table(
                pd.DataFrame(
                    {
                        "Field": ["Age", "Annual Income", "Employment Length", "Home Ownership"],
                        "Value": [
                            f"{age} yrs",
                            f"${income:,}",
                            f"{emp_length} yrs",
                            home_ownership,
                        ],
                    }
                ).set_index("Field")
            )

        with right:
            st.markdown("**Loan Details**")
            st.table(
                pd.DataFrame(
                    {
                        "Field": [
                            "Loan Amount", "Interest Rate", "Loan Intent",
                            "Loan Grade", "Credit History", "Prior Default",
                        ],
                        "Value": [
                            f"${loan_amount:,}",
                            f"{interest_rate:.1f}%",
                            loan_intent.title(),
                            loan_grade,
                            f"{cred_hist_length} yrs",
                            historical_default,
                        ],
                    }
                ).set_index("Field")
            )

    # Row 5 – Raw feature vector (collapsible)
    with st.expander("🔬 Model Input Feature Vector", expanded=False):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
