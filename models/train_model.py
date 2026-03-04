from pathlib import Path
import sys
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.feature_engineering import load_transactions, engineer_features, feature_columns


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-6)


def make_synthetic_label(df_feats: pd.DataFrame) -> pd.Series:
    sav = _minmax(df_feats['savings_ratio'].clip(-1, 1))
    net = _minmax(df_feats['net_cashflow'])
    cf = df_feats['cashflow_stability'].clip(0, 1)
    ic = df_feats['income_consistency'].clip(0, 1)
    spend = 1 - _minmax(df_feats['spending_ratio'].clip(lower=0, upper=10))
    vol = 1 - _minmax(df_feats['amount_volatility'])
    incv = 1 - _minmax(df_feats['income_volatility'])
    score = 0.25 * sav + 0.2 * net + 0.15 * cf + 0.15 * ic + 0.1 * spend + 0.08 * vol + 0.07 * incv
    thresh = score.median()
    y = (score > thresh).astype(int).rename('creditworthy')
    return y


def risk_level(trust_score: float) -> str:
    if trust_score >= 70:
        return 'Low'
    if trust_score >= 40:
        return 'Medium'
    return 'High'


def fit(csv_path='data/transactions.csv', model_path='models/model.joblib', predictions_out='models/predictions.csv'):
    root = Path(__file__).resolve().parents[1]
    data_path = (root / csv_path).resolve()
    model_out = (root / model_path).resolve()
    preds_out = (root / predictions_out).resolve()
    df = load_transactions(str(data_path))
    feats = engineer_features(df)
    cols = feature_columns(feats)
    X = feats[cols].copy()
    y = make_synthetic_label(feats)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    clf.fit(X_train, y_train)
    proba_test = clf.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, proba_test)
    except Exception:
        auc = float('nan')
    full_proba = clf.predict_proba(X_imp)[:, 1]
    trust_scores = (100 * full_proba).round(2)
    risk = [risk_level(s) for s in trust_scores]
    out_df = pd.DataFrame({
        'user_id': feats['user_id'],
        'creditworthy': y.values,
        'TrustScore': trust_scores,
        'Risk Level': risk
    })
    model_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            'model': clf,
            'imputer': imputer,
            'feature_columns': cols,
            'feature_importances': {c: float(v) for c, v in zip(cols, clf.feature_importances_.tolist())},
            'trained_at': int(time.time()),
            'auc_test': auc
        },
        str(model_out)
    )
    out_df.to_csv(str(preds_out), index=False)
    print(str(model_out))
    print(str(preds_out))
    print(f'Test AUC: {auc}')
    return str(model_out), str(preds_out), auc


if __name__ == '__main__':
    fit()
