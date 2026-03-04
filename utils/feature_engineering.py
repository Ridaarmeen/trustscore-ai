import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime


def load_transactions(path):
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()
    if 'timestamp' in df.columns and not is_datetime(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['date'] = df['timestamp'] if 'timestamp' in df.columns else pd.NaT
    if 'transaction_amount' in df.columns:
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce')
    if 'transaction_type' in df.columns:
        inc = df['transaction_type'].astype(str).str.lower().eq('income')
        exp = df['transaction_type'].astype(str).str.lower().eq('expense')
        df['amount_signed'] = np.where(inc, df['transaction_amount'], np.where(exp, -df['transaction_amount'], np.nan))
    subset_cols = [c for c in ['user_id', 'date', 'transaction_amount', 'transaction_type'] if c in df.columns]
    df = df.dropna(subset=subset_cols)
    return df


def _weekly_series(group, value_col, only_income=False):
    x = group.copy()
    if only_income and 'transaction_type' in x.columns:
        x = x[x['transaction_type'].astype(str).str.lower() == 'income']
    if x.empty:
        return pd.Series(dtype=float)
    w = x['date'].dt.to_period('W').dt.to_timestamp()
    s = x.groupby(w)[value_col].sum().sort_index()
    return s


def engineer_features(df):
    df = preprocess(df)
    df = df.sort_values(['user_id', 'date'])
    g = df.groupby('user_id', dropna=False)
    total_income = g.apply(lambda x: x.loc[x['transaction_type'].astype(str).str.lower() == 'income', 'transaction_amount'].sum()).rename('total_income')
    total_expense = g.apply(lambda x: x.loc[x['transaction_type'].astype(str).str.lower() == 'expense', 'transaction_amount'].sum()).rename('total_expense')
    tx_count = g.size().rename('transaction_count')
    avg_tx_amt = g['transaction_amount'].apply(lambda s: s.abs().mean()).rename('average_transaction_amount')
    vol_amt = g['amount_signed'].std().rename('amount_volatility')
    last_date = g['date'].max().rename('last_tx_date')
    max_date = df['date'].max()
    days_since_last = (max_date - last_date).dt.days.rename('days_since_last_tx')
    def freq_per_week(x):
        n = x.shape[0]
        d = (x['date'].max() - x['date'].min()).days + 1
        weeks = max(d / 7.0, 1.0 / 7.0)
        return n / weeks
    tx_freq = g.apply(freq_per_week).rename('transaction_frequency')
    def inc_metrics(x):
        s = _weekly_series(x, 'transaction_amount', only_income=True)
        if s.empty:
            return pd.Series({'income_consistency': 0.0, 'income_volatility': 0.0})
        m = s.mean()
        sd = s.std(ddof=0) if len(s) > 1 else 0.0
        cv = sd / (m + 1e-6)
        return pd.Series({'income_consistency': 1.0 / (1.0 + cv), 'income_volatility': cv})
    inc_df = g.apply(inc_metrics)
    def cashflow_metric(x):
        s = _weekly_series(x, 'amount_signed', only_income=False)
        if s.empty:
            return 0.0
        m = np.abs(s.mean())
        sd = s.std(ddof=0) if len(s) > 1 else 0.0
        cv = sd / (m + 1e-6)
        return 1.0 / (1.0 + cv)
    cfs = g.apply(lambda x: pd.Series({'cashflow_stability': cashflow_metric(x)}))
    res = pd.concat([total_income, total_expense, tx_freq, avg_tx_amt, inc_df, cfs, tx_count, vol_amt, last_date, days_since_last], axis=1)
    res['spending_ratio'] = res['total_expense'] / (res['total_income'] + 1e-6)
    res['savings_ratio'] = (res['total_income'] - res['total_expense']) / (res['total_income'] + 1e-6)
    res['net_cashflow'] = res['total_income'] - res['total_expense']
    res = res.fillna(0)
    res = res.reset_index()
    return res[['user_id', 'total_income', 'total_expense', 'spending_ratio', 'savings_ratio', 'transaction_frequency', 'average_transaction_amount', 'income_consistency', 'income_volatility', 'cashflow_stability', 'net_cashflow', 'amount_volatility', 'days_since_last_tx']]


def feature_columns(df_features):
    cols = [c for c in df_features.columns if c not in ['user_id', 'last_tx_date']]
    return cols


def recommend_loan(trust_score: float, monthly_income: float, emi_ratio: float = 0.3):
    ts = float(trust_score)
    mi = max(float(monthly_income), 0.0)
    if ts > 80:
        loan = 50000
        band = '80+'
    elif ts >= 60:
        loan = 25000
        band = '60–80'
    elif ts >= 40:
        loan = 10000
        band = '40–60'
    else:
        loan = 0
        band = '<40'
    safe_emi = round(mi * float(emi_ratio), 2)
    return {
        'loan_eligibility': loan,
        'safe_emi': safe_emi,
        'trust_score': ts,
        'trust_score_band': band,
        'recommended': loan > 0
    }
