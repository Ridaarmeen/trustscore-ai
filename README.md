# TrustScore AI

AI-powered alternative credit scoring for gig workers using transaction behavior.

## Tech Stack
- Python, Pandas, scikit-learn
- Streamlit, Plotly

## Project Structure
```
TrustScore AI/
├── data/
│   └── transactions.csv
├── models/
│   └── train_model.py
├── utils/
│   └── feature_engineering.py
├── app/
│   └── streamlit_app.py
└── requirements.txt
```

## Setup
1) Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```
pip install -r requirements.txt
```

## Data Schema
Sample CSV columns:
- user_id, date, amount, merchant_category, tx_type, platform, successful

## Train Model
Run from project root:
```
python models/train_model.py
```
This saves models/model.joblib.

## Run App
Run from project root:
```
streamlit run app/streamlit_app.py
```
Upload your transactions CSV or use the sample. The app shows per-worker features and TrustScore distribution.

## Notes
- The starter model clusters behavior and derives a relative TrustScore without labels.
- Replace sample data with your own and retrain for better results.
