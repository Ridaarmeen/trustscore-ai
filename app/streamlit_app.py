import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="TrustScore AI", layout="wide")

st.title("TrustScore AI")
st.write("AI-powered alternative credit scoring for gig workers")

st.markdown("---")

uploaded_file = st.file_uploader("Upload transaction CSV")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Transaction Data Preview")
    st.dataframe(df.head())

    st.markdown("---")

    # -----------------------
    # Financial Calculations
    # -----------------------

    total_income = df[df["transaction_type"] == "income"]["transaction_amount"].sum()
    total_expense = df[df["transaction_type"] == "expense"]["transaction_amount"].sum()

    savings = total_income - total_expense
    spending_ratio = total_expense / total_income if total_income > 0 else 0

    # -----------------------
    # Financial Metrics
    # -----------------------

    st.subheader("Financial Behavior Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Income", f"${total_income:,.2f}")
    col2.metric("Total Expense", f"${total_expense:,.2f}")
    col3.metric("Savings", f"${savings:,.2f}")
    col4.metric("Spending Ratio", round(spending_ratio,2))

    st.markdown("---")

    # -----------------------
    # TrustScore Calculation
    # -----------------------

    trust_score = max(0, min(100, int((1 - spending_ratio) * 100)))

    st.subheader("TrustScore")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=trust_score,
        title={'text': "TrustScore"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0,50], 'color': "red"},
                {'range': [50,75], 'color': "yellow"},
                {'range': [75,100], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # -----------------------
    # Risk Level
    # -----------------------

    if trust_score > 75:
        risk = "Low Risk"
    elif trust_score > 50:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    st.write("Risk Level:", risk)

    # -----------------------
    # Financial Health
    # -----------------------

    st.subheader("Financial Health")

    if trust_score > 75:
        st.success("Healthy Financial Behavior")
    elif trust_score > 50:
        st.warning("Moderate Financial Risk")
    else:
        st.error("High Financial Risk")

    st.markdown("---")

    # -----------------------
    # Loan Recommendation
    # -----------------------

    st.subheader("Loan Recommendation")

    if trust_score > 80:
        loan = "$50,000"
    elif trust_score > 60:
        loan = "$25,000"
    elif trust_score > 40:
        loan = "$10,000"
    else:
        loan = "Not eligible"

    st.write("Eligible Loan Amount:", loan)

    st.markdown("---")

    # -----------------------
    # Income vs Expense Chart
    # -----------------------

    st.subheader("Income vs Expense")

    chart_data = pd.DataFrame({
        "Category": ["Income", "Expense"],
        "Amount": [total_income, total_expense]
    })

    st.bar_chart(chart_data.set_index("Category"))

    st.markdown("---")

    # -----------------------
    # Financial Simulation
    # -----------------------

    st.subheader("Financial Simulation")

    st.write("Simulate how income or expenses affect your TrustScore")

    income_change = st.slider("Change Income (%)", -50, 50, 0)
    expense_change = st.slider("Change Expense (%)", -50, 50, 0)

    sim_income = total_income * (1 + income_change / 100)
    sim_expense = total_expense * (1 + expense_change / 100)

    sim_spending_ratio = sim_expense / sim_income if sim_income > 0 else 0
    sim_trust_score = max(0, min(100, int((1 - sim_spending_ratio) * 100)))

    st.metric("Simulated TrustScore", sim_trust_score)

    st.markdown("---")

    # -----------------------
    # AI Financial Insights
    # -----------------------

    st.subheader("AI Financial Insights")

    if spending_ratio < 0.5:
        st.success("Great financial discipline! Your expenses are well controlled.")
    elif spending_ratio < 0.8:
        st.warning("Your spending is moderate. Increasing savings could improve your TrustScore.")
    else:
        st.error("High spending detected. Reducing expenses can significantly improve your TrustScore.")