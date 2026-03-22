"""
preprocessing.py
----------------
Data loading, cleaning, feature engineering, encoding and splitting.

Dataset : IBM Telco Customer Churn (Kaggle — free, CC BY 1.0)
URL     : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
File    : WA_Fn-UseC_-Telco-Customer-Churn.csv  →  place in  data/raw/
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET    = "Churn"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

DROP_COLS = ["customerID"]


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the IBM Telco CSV from disk.

    How to get the file:
      1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
      2. Download  WA_Fn-UseC_-Telco-Customer-Churn.csv
      3. Move it to  data/raw/
    """
    df = pd.read_csv(path)
    print(f"[load]  {df.shape[0]:,} rows  ×  {df.shape[1]} cols")
    return df


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop customerID (identifier, not a feature).
    - Fix TotalCharges: stored as string with spaces where values are missing.
    - Impute the ~11 missing TotalCharges rows with the column median.
    - Binary-encode the target: Yes → 1, No → 0.
    - SeniorCitizen is already 0/1, kept as numeric.
    """
    df = df.copy()

    # Drop identifier
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # TotalCharges: coerce spaces → NaN, then impute with median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    churn_rate = df[TARGET].mean()
    print(f"[clean] churn rate = {churn_rate:.1%}  |  shape = {df.shape}")
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified 80/20 split — preserves the ~26 % churn rate in both sets."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    print(f"[split] train={train.shape}  test={test.shape}")
    return train, test


# ── Encode & scale ────────────────────────────────────────────────────────────

def fit_transform(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    encoder_path: str = None,
    scaler_path:  str = None,
):
    """
    1. One-hot encode categorical columns (fit on train only).
    2. Standard-scale continuous columns  (fit on train only).
    3. Optionally persist encoder and scaler to disk.

    Returns
    -------
    X_train, y_train, X_test, y_test, encoder, scaler, encoded_feature_names
    """
    X_train = train_df.drop(columns=[TARGET]).copy()
    y_train = train_df[TARGET].copy()
    X_test  = test_df.drop(columns=[TARGET]).copy()
    y_test  = test_df[TARGET].copy()

    # One-hot encoding
    enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    enc.fit(X_train[CATEGORICAL_COLS])
    enc_feats = list(enc.get_feature_names_out(CATEGORICAL_COLS))

    X_train[enc_feats] = enc.transform(X_train[CATEGORICAL_COLS])
    X_test[enc_feats]  = enc.transform(X_test[CATEGORICAL_COLS])
    X_train.drop(columns=CATEGORICAL_COLS, inplace=True)
    X_test.drop(columns=CATEGORICAL_COLS, inplace=True)

    # Standard scaling
    scaler = StandardScaler()
    X_train[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test[CONTINUOUS_COLS]  = scaler.transform(X_test[CONTINUOUS_COLS])

    # Persist artefacts
    if encoder_path:
        dump(enc, open(encoder_path, "wb"))
        print(f"[prep]  encoder  → {encoder_path}")
    if scaler_path:
        dump(scaler, open(scaler_path, "wb"))
        print(f"[prep]  scaler   → {scaler_path}")

    print(f"[prep]  X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, y_train, X_test, y_test, enc, scaler, enc_feats


def preprocess_new(data: pd.DataFrame, enc, scaler, enc_feats: list) -> pd.DataFrame:
    """Apply already-fitted encoder + scaler to new (unseen) data."""
    df = data.copy()
    df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df[enc_feats] = enc.transform(df[CATEGORICAL_COLS])
    df.drop(columns=CATEGORICAL_COLS, inplace=True)
    df[CONTINUOUS_COLS] = scaler.transform(df[CONTINUOUS_COLS])
    return df


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
    print("\n✅  Preprocessing done.")
