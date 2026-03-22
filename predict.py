"""
predict.py
----------
Load the saved XGBoost model + artefacts and run predictions on new data.

Usage
-----
python src/predict.py --input data/raw/new_customers.csv --output data/processed/predictions.csv
"""

import argparse
import pandas as pd
import xgboost as xgb
from pickle import load

from preprocessing import preprocess_new, CATEGORICAL_COLS, CONTINUOUS_COLS


def run(input_path: str, output_path: str,
        model_path:   str = "models/xgb_model.json",
        encoder_path: str = "models/encoder.pkl",
        scaler_path:  str = "models/scaler.pkl",
        threshold:    float = 0.5):

    # Load artefacts
    model   = xgb.Booster(); model.load_model(model_path)
    encoder = load(open(encoder_path, "rb"))
    scaler  = load(open(scaler_path,  "rb"))
    enc_feats = list(encoder.get_feature_names_out(CATEGORICAL_COLS))

    # Load & preprocess new data
    raw = pd.read_csv(input_path)
    X   = preprocess_new(raw, encoder, scaler, enc_feats)

    # Predict
    proba = model.predict(xgb.DMatrix(X))
    preds = (proba >= threshold).astype(int)

    # Build output
    out = raw[["customerID"]].copy() if "customerID" in raw.columns else raw.iloc[:, :1].copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"]  = preds
    out["churn_label"]       = out["churn_prediction"].map({1: "Churn", 0: "No Churn"})

    out.to_csv(output_path, index=False)
    print(f"[predict]  {len(out):,} customers  →  {output_path}")
    print(f"[predict]  predicted churners: {preds.sum()} ({preds.mean():.1%})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn prediction inference")
    parser.add_argument("--input",   required=True,  help="Path to new customer CSV")
    parser.add_argument("--output",  required=True,  help="Path to save predictions CSV")
    parser.add_argument("--model",   default="models/xgb_model.json")
    parser.add_argument("--encoder", default="models/encoder.pkl")
    parser.add_argument("--scaler",  default="models/scaler.pkl")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run(args.input, args.output, args.model, args.encoder, args.scaler, args.threshold)
