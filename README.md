# Fraud Detection — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?style=flat-square)
![SageMaker](https://img.shields.io/badge/AWS-SageMaker-yellow?style=flat-square&logo=amazonaws)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

**[→ Live Project Showcase](https://ummuglsm0134.github.io/fraud-detection/fraud_detection_showcase.html)**

**Dataset:** IEEE-CIS Fraud Detection  590k transactions · 3.5% fraud rate · 27:1 class imbalance 
**Stack:** Python · XGBoost · Scikit-learn · Pandas · AWS SageMaker

---

## Why This Project

At GVTC Communications I shipped a production customer churn classifier (AUC ~0.97+)
adopted into VP level retention strategy behavioral feature engineering on transactional
data, ensemble classifiers on imbalanced targets, and business-cost-aware threshold tuning.

Fraud detection is that same problem with higher stakes: missed detections cost real money,
fraud patterns are adversarial, and real time serving adds an infrastructure dimension.
This project covers the full stack from raw data to SageMaker endpoint.

---
# Data

Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data

Place `train_transaction.csv` and `train_identity.csv` here.

> Not required — notebooks generate synthetic data automatically if files are missing.

## Results

| Model | PR-AUC | ROC-AUC | Recall | Precision |
|-------|--------|---------|--------|-----------|
| Logistic Regression | 0.62 | 0.88 | 0.71 | 0.58 |
| Random Forest | 0.78 | 0.94 | 0.76 | 0.72 |
| **XGBoost** | **0.84** | **0.97+** | **0.81** | **0.76** |

> **Primary metric is PR-AUC**, not ROC-AUC. On a 3.5% fraud rate,
> ROC-AUC is misleading — a model predicting "never fraud" on everything
> still scores well. Precision-Recall AUC focuses entirely on the class that matters.

---

## Key Decisions at a Glance

| Decision | Choice | Why |
|----------|--------|-----|
| Primary metric | PR-AUC | ROC-AUC misleads on 3.5% fraud rate |
| Imbalance handling | `scale_pos_weight=27.6` | Corrects loss function directly |
| Threshold | 0.35 (not 0.5) | F-beta (β=2) weights recall 2× over precision |
| Model | XGBoost | Best PR-AUC; handles feature interactions natively |
| Deployment | SageMaker endpoint | REST API, versioned artifacts, Model Monitor |
---

## Key Engineering Decisions

### 1. Class Imbalance — Three-Layer Approach
```python
# Layer 1: Weighted loss function
xgb_model = xgb.XGBClassifier(scale_pos_weight=27.6)

# Layer 2: Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Layer 3: Threshold at 0.35, not default 0.5
optimal_threshold = 0.35  # tuned to maximize F-beta (β=2)
```

### 2. Behavioral Feature Engineering
```python
# Deviation from card's own baseline — $500 means different things per customer
df['amt_zscore'] = (df['TransactionAmt'] - df['card_mean_amt']) / df['card_std_amt']
df['amt_to_card_median_ratio'] = df['TransactionAmt'] / df['card_median_amt']

# Per-customer median imputation (not global median)
df['TransactionAmt'] = df.groupby('card1')['TransactionAmt'].transform(
    lambda x: x.fillna(x.median())
)

# Missing distance is itself a fraud signal — don't just impute it away
df['dist1_missing'] = df['dist1'].isna().astype(int)
```

### 3. Threshold Optimization for Business Cost
```python
# F-beta (β=2): recall weighted 2× — missed fraud costs more than a false alarm
beta = 2
fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
optimal_threshold = thresholds[np.argmax(fbetas)]
```

### 4. SageMaker Inference Response
```python
# inference.py returns structured fraud assessment
{
    "fraud_score":  0.82,
    "is_fraud":     True,
    "risk_level":   "HIGH"   # HIGH / MEDIUM / LOW
}
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/ummuglsm0134/fraud-detection
cd fraud-detection
pip install -r requirements.txt

# 2. Get data (Kaggle account required)
# https://www.kaggle.com/c/ieee-fraud-detection/data
# Place train_transaction.csv and train_identity.csv in data/

# 3. Run notebooks in order
jupyter notebook notebooks/01_EDA_Feature_Engineering.ipynb
jupyter notebook notebooks/02_Modeling_Evaluation.ipynb
jupyter notebook notebooks/03_SageMaker_Deployment.ipynb
```

> **No Kaggle account?** All notebooks automatically generate a synthetic
> dataset and run end-to-end without any data download.

---

## Author

**Lia Arslan** — Data Scientist · Boston, MA  
M.S. Statistics & Data Science · 6+ years across telecom, healthcare, clean energy, real estate  
[linkedin.com/in/uarslan](https://linkedin.com/in/uarslan) · [github.com/ummuglsm0134](https://github.com/ummuglsm0134)

---

## Repository Structure
