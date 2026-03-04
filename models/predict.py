import numpy as np
import pandas as pd


def risk_level(score: float) -> str:
    if score >= 70:
        return 'Low'
    if score >= 40:
        return 'Medium'
    return 'High'


def predict_scores(df_features: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    df = df_features.copy()
    cols = artifact['feature_columns']
    X = df[cols].copy()
    X_imp = artifact['imputer'].transform(X)
    proba = artifact['model'].predict_proba(X_imp)[:, 1]
    trust = (100 * proba).round(2)
    df['trust_score'] = trust
    df['risk_level'] = [risk_level(float(s)) for s in trust]
    return df
