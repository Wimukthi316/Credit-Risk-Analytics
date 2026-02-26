<div align="center">

# 🏦 Credit Risk Prediction Dashboard

### AI-powered loan default risk assessment with real-time scoring and financial loss estimation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-risk-analytics-whgaxsl4zbcsf9vfmdjpks.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)

</div>

---

## 📌 Project Overview

This project delivers a production-grade **Credit Risk Prediction Dashboard** built on a **Random Forest classifier** trained on 32,000+ real-world loan records. Given a loan applicant's personal and financial profile, the system:

- Predicts the **Probability of Default (PD)** — the likelihood the borrower will fail to repay
- Calculates the **Expected Financial Loss (EFL)** — quantifying the actual monetary exposure: `EFL = PD × Loan Amount`
- Renders an instant **Approve / Review / Reject** underwriting decision based on configurable risk thresholds
- Explains every result through a clean, interactive UI accessible to both technical and non-technical users

This mirrors a real-world credit decisioning engine that a bank's automated underwriting system would employ.

---

## � Project Showcase

| ✅ Low Risk — Approve | ❌ High Risk — Reject |
|:---:|:---:|
| ![Approve Screenshot](assets/approve.png) | ![Reject Screenshot](assets/reject.png) |

---

## �🚀 Live Demo

> **[→ Open the live app on Streamlit Cloud](https://credit-risk-analytics-whgaxsl4zbcsf9vfmdjpks.streamlit.app/)**

On first load, the app trains the model directly from the raw dataset (~30 seconds). All subsequent interactions are cached and instant.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **End-to-End ML Pipeline** | Data cleaning → feature engineering → ordinal + one-hot encoding → Random Forest training, all reproduced automatically at runtime |
| **Probability of Default** | Calibrated class probability output from `predict_proba`, not just a binary yes/no prediction |
| **Expected Financial Loss** | Monetises risk: `P(default) × Loan Amount`, giving loan officers a concrete ₹ figure at stake |
| **Three-Tier Decision Engine** | `✅ APPROVE` (PD < 30%) · `⚠️ REVIEW` (30–60%) · `❌ REJECT` (> 60%) with written rationale per decision |
| **Interactive Risk Meter** | Colour-coded progress bar (green → amber → red) that moves with the prediction |
| **Professional UI** | Custom CSS theming, KPI cards, responsive two-column layout, collapsible application summary |
| **Zero-Setup Deployment** | Model trains from raw CSV on first launch — no pre-built `.pkl` required on the server |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (Random Forest Classifier) |
| **Web App / UI** | Streamlit |
| **Visualisation** | Matplotlib, Seaborn, Altair |
| **Explainability** | SHAP (TreeExplainer + Waterfall plots) |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Project Structure

```
credit-risk-analytics/
│
├── app/
│   └── app.py                             # Streamlit dashboard
│
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv        # Original dataset (32,582 records)
│   └── processed/
│       └── processed_credit_risk_dataset.csv
│
├── models/                                # Auto-created at runtime (git-ignored)
│   └── credit_risk_model.pkl
│
├── notebooks/
│   └── credit_risk.ipynb                  # EDA, training experiments, SHAP analysis
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model & Feature Engineering

The pipeline exactly mirrors the notebook experiments:

**Preprocessing steps:**
- Remove age outliers (`person_age > 100`) and employment length outliers (`> 100 years`)
- Impute missing `person_emp_length` and `loan_int_rate` with their respective medians
- Ordinal-encode `loan_grade`: `A=0, B=1, C=2, D=3, E=4, F=5, G=6`
- Binary-encode `cb_person_default_on_file`: `Y→1, N→0`
- One-hot encode `person_home_ownership` (reference: `MORTGAGE`) and `loan_intent` (reference: `DEBTCONSOLIDATION`) with `drop_first=True`
- Derive `loan_percent_income = loan_amnt / person_income`

**Final feature set (17 features):**

```
person_age · person_income · person_emp_length · loan_grade · loan_amnt ·
loan_int_rate · loan_percent_income · cb_person_default_on_file ·
cb_person_cred_hist_length · person_home_ownership_OTHER/OWN/RENT ·
loan_intent_EDUCATION/HOMEIMPROVEMENT/MEDICAL/PERSONAL/VENTURE
```

**Model:** `RandomForestClassifier(n_estimators=100, random_state=42)` — 80/20 train-test split

---

## 🖥 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/credit-risk-analytics.git
cd credit-risk-analytics
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the dashboard**
```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The model will train automatically on first run.

---

## 📊 Dataset

- **Source:** [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Records:** 32,581 loan applications
- **Target:** `loan_status` — `0` = Repaid · `1` = Defaulted
- **Class distribution:** ~78% repaid / ~22% defaulted

---

## 📈 Results

| Metric | Value |
|---|---|
| Algorithm | Random Forest (100 estimators) |
| Train / Test Split | 80% / 20% |
| Accuracy | ~93% |
| Explainability | SHAP TreeExplainer + Waterfall chart |

> Full classification report and SHAP feature importance plots are available in [`notebooks/credit_risk.ipynb`](notebooks/credit_risk.ipynb).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---