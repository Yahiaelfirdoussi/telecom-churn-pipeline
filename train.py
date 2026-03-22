"""
train.py
--------
Train Logistic Regression, Random Forest, and XGBoost on the Telco churn data.
Prints a side-by-side comparison table and saves the best model (XGBoost).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

from preprocessing import load_data, clean_data, split_data, fit_transform


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name: str, model, X_tr, y_tr, X_te, y_te, is_xgb=False) -> dict:
    if is_xgb:
        y_tr_pred = (model.predict(xgb.DMatrix(X_tr)) >= 0.5).astype(int)
        y_te_pred = (model.predict(xgb.DMatrix(X_te)) >= 0.5).astype(int)
    else:
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

    res = dict(
        Model        = name,
        Train_F1     = round(f1_score(y_tr, y_tr_pred), 4),
        Train_Recall = round(recall_score(y_tr, y_tr_pred), 4),
        Train_AUC    = round(roc_auc_score(y_tr, y_tr_pred), 4),
        Test_F1      = round(f1_score(y_te, y_te_pred), 4),
        Test_Recall  = round(recall_score(y_te, y_te_pred), 4),
        Test_Prec    = round(precision_score(y_te, y_te_pred), 4),
        Test_AUC     = round(roc_auc_score(y_te, y_te_pred), 4),
    )

    sep = "=" * 55
    print(f"\n{sep}\n  {name}\n{sep}")
    print(f"  Train →  F1 {res['Train_F1']}  |  Recall {res['Train_Recall']}  |  AUC {res['Train_AUC']}")
    print(f"  Test  →  F1 {res['Test_F1']}  |  Recall {res['Test_Recall']}  |  Prec {res['Test_Prec']}  |  AUC {res['Test_AUC']}")
    print(f"  Confusion matrix (test):\n{confusion_matrix(y_te, y_te_pred)}")
    return res


# ── Train all models ──────────────────────────────────────────────────────────

def train_all(X_train, y_train, X_test, y_test):
    """
    Train three classifiers and return a comparison DataFrame + the XGBoost model.
    Class imbalance (~26 % churn) is handled via class weights / scale_pos_weight.
    """
    neg, pos  = (y_train == 0).sum(), (y_train == 1).sum()
    spw       = neg / pos                  # scale_pos_weight for XGBoost
    cw        = {0: 1.0, 1: spw}           # class_weight for sklearn

    results = []

    # 1 ── Logistic Regression
    lr = LogisticRegression(
        C=0.5, max_iter=1000, class_weight=cw, random_state=42
    )
    lr.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression", lr, X_train, y_train, X_test, y_test))

    # 2 ── Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight=cw,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test))

    # 3 ── XGBoost
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "scale_pos_weight": spw,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "seed":             42,
    }
    dtrain    = xgb.DMatrix(X_train, label=y_train)
    dtest     = xgb.DMatrix(X_test,  label=y_test)
    xgb_model = xgb.train(
        params, dtrain,
        num_boost_round=300,
        evals=[(dtest, "eval")],
        early_stopping_rounds=20,
        verbose_eval=50,
    )
    results.append(evaluate(
        "XGBoost", xgb_model, X_train, y_train, X_test, y_test, is_xgb=True
    ))

    # Comparison table
    comp = pd.DataFrame(results)
    print("\n\n📊  Model comparison")
    print(comp.to_string(index=False))
    return comp, lr, rf, xgb_model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw              = load_data()
    clean            = clean_data(raw)
    train_df, test_df = split_data(clean)

    X_train, y_train, X_test, y_test, enc, scaler, enc_feats = fit_transform(
        train_df, test_df,
        encoder_path="models/encoder.pkl",
        scaler_path="models/scaler.pkl",
    )

    comp, lr, rf, xgb_model = train_all(X_train, y_train, X_test, y_test)

    xgb_model.save_model("models/xgb_model.json")
    print("\n[save]  XGBoost  →  models/xgb_model.json")
    print("✅  Training done.")
