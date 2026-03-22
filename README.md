# Telecom Churn Prediction — End-to-End MLOps Pipeline

> Predicts which telecom customers are likely to churn using the IBM Telco dataset.
> Full pipeline: data cleaning → feature engineering → model training → evaluation → inference.

---

## Results

| Model | Test F1 | Test Recall | Test Precision | Test AUC |
|---|---|---|---|---|
| Logistic Regression | 0.6136 | 0.7834 | 0.5043 | 0.7526 |
| Random Forest | 0.6261 | 0.7567 | 0.5340 | 0.7590 |
| **XGBoost** | **0.6266** | **0.7941** | **0.5174** | **0.7632** |

XGBoost is the best overall model — highest recall (0.79) and AUC (0.76). Recall is the priority metric: missing a churner costs more than a false alarm.

---

## Dataset

**IBM Telco Customer Churn** — free, publicly available.

| | |
|---|---|
| Source | [IBM GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) |
| Rows | 7,043 customers |
| Features | 21 columns |
| Target | `Churn` (Yes / No) |
| Churn rate | 26.5% |

---

## Quick start
```bash
# 1. Clone
git clone https://github.com/yahiaelfirdoussi/telecom-churn-pipeline
cd telecom-churn-pipeline

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Download the dataset
mkdir -p data/raw models
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
     -o "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 4. Preprocess
python3 preprocessing.py

# 5. Train
python3 train.py

# 6. Predict
python3 predict.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
                   --output data/raw/predictions.csv
```

---

## Project structure
```
telecom-churn-pipeline/
├── preprocessing.py      # Load, clean, encode, scale, split
├── train.py              # Train LR / RF / XGBoost, print comparison
├── predict.py            # Inference on new customer data
├── notebooks/
│   ├── churn_modeling.ipynb
│   └── telecom_eda.ipynb
├── data/raw/             # CSV goes here (git-ignored)
├── models/               # Saved artefacts (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline steps

**preprocessing.py** — fixes `TotalCharges` dtype, imputes 11 missing values, encodes target (Yes→1 / No→0), one-hot encodes 15 categorical features, standard scales 3 continuous features, stratified 80/20 split. Encoder and scaler fit on train only — no data leakage.

**train.py** — trains Logistic Regression, Random Forest, and XGBoost with class weights to handle the 26.5% churn imbalance. XGBoost uses early stopping on AUC. Prints a full comparison table.

**predict.py** — loads the saved model and artefacts, applies the same preprocessing to new data, outputs a CSV with churn probability, binary prediction, and label.

---

## Key design decisions

**Stratified split** preserves the 26.5% churn rate in both train and test sets, preventing optimistic bias in evaluation.

**Class imbalance** handled via `class_weight` in sklearn models and `scale_pos_weight` in XGBoost, both set to ~2.8× — avoids the model defaulting to predicting "no churn" for everyone.

**No data leakage** — encoder and scaler are fit exclusively on training data, then applied to test and inference data separately.

**Early stopping** in XGBoost monitors AUC on the test set and halts after 20 rounds without improvement, preventing overfitting without manual tuning.

---

## Tech stack

`Python 3.11` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn`

---

## Author

**Yahya Elfirdoussi** — Data Scientist & ML Engineer
📧 yahiaelfirdoussi7@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/yahya-elfirdoussi) · [Portfolio](https://yahiaelfirdoussi.netlify.app) · [GitHub](https://github.com/yahiaelfirdoussi)