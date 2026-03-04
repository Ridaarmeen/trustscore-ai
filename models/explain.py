import pandas as pd
import numpy as np


def get_feature_importance(artifact):
    if artifact is None:
        return pd.DataFrame(columns=['feature', 'importance', 'importance_pct'])
    cols = artifact.get('feature_columns', [])
    imp_map = artifact.get('feature_importances', None)
    if imp_map is not None:
        items = [(k, float(v)) for k, v in imp_map.items()]
    else:
        model = artifact.get('model', None)
        if model is None or not hasattr(model, 'feature_importances_'):
            return pd.DataFrame(columns=['feature', 'importance', 'importance_pct'])
        arr = np.asarray(model.feature_importances_, dtype=float)
        items = list(zip(cols, arr.tolist()))
    df = pd.DataFrame(items, columns=['feature', 'importance'])
    total = df['importance'].sum() + 1e-12
    df['importance_pct'] = 100.0 * df['importance'] / total
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def top_n_importance(artifact, n=12):
    df = get_feature_importance(artifact)
    if df.empty:
        return df
    return df.head(n)
