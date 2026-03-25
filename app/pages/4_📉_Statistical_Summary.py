import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_engine import predict_future, detect_anomalies
from ai_engine import generate_insight, generate_pdf

st.set_page_config(page_title="Statistical Summary", page_icon="📉", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    .analysis-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .prediction-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0.5rem 0;
    }

    .action-button {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
    }

    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(56, 189, 248, 0.4);
    }

    h1, h2, h3 { color: #38bdf8; }

    .stDataFrame {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📉 Statistical Summary")
st.caption("Predictions, Anomalies & Advanced Analytics")

if 'uploaded_data' not in st.session_state:
    st.warning("Please upload a dataset on the Home page first")
    st.stop()

df = st.session_state['uploaded_data']
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found for statistical analysis")
    st.stop()

st.markdown("---")

selected_column = st.selectbox("Select a column for analysis", numeric_cols)

st.markdown("### Analysis Operations")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Predict Future Values", use_container_width=True):
        with st.spinner("Running prediction model..."):
            predictions = predict_future(df, selected_column, steps=5)
            st.session_state['predictions'] = predictions
            st.session_state['pred_column'] = selected_column

with col2:
    if st.button("Detect Anomalies", use_container_width=True):
        with st.spinner("Detecting anomalies..."):
            anomalies = detect_anomalies(df, selected_column)
            st.session_state['anomalies'] = anomalies
            st.session_state['anom_column'] = selected_column

with col3:
    if st.button("Generate AI Insight", use_container_width=True):
        with st.spinner("Generating insights..."):
            insight = generate_insight(df, selected_column)
            st.session_state['insight'] = insight
            st.session_state['insight_column'] = selected_column

with col4:
    if st.button("Export Full Report", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            preds = st.session_state.get('predictions', 'No predictions generated')
            anomalies = st.session_state.get('anomalies', 'No anomalies detected')
            insights = st.session_state.get('insight', 'No insights generated')

            generate_pdf("Intellectual_Analysis_Report.pdf", insights, preds, anomalies)

            with open("Intellectual_Analysis_Report.pdf", "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="Intellectual_Analysis_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

st.markdown("---")

if 'predictions' in st.session_state:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown(f"### Future Value Predictions for '{st.session_state['pred_column']}'")

    predictions = st.session_state['predictions']

    col1, col2 = st.columns([2, 1])

    with col1:
        historical_values = df[st.session_state['pred_column']].values
        future_indices = list(range(len(historical_values), len(historical_values) + len(predictions)))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(len(historical_values))),
            y=historical_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#38bdf8', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=future_indices,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='#818cf8', width=2, dash='dash'),
            marker=dict(size=10)
        ))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            hovermode='x unified',
            title="Prediction Forecast"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Predicted Values")
        for i, pred in enumerate(predictions, 1):
            st.markdown(f'<div class="prediction-value">Step {i}: {pred:.2f}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if 'anomalies' in st.session_state:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown(f"### Anomaly Detection for '{st.session_state['anom_column']}'")

    anomalies = st.session_state['anomalies']

    if len(anomalies) > 0:
        st.warning(f"Found {len(anomalies)} anomalous data points")
        st.dataframe(anomalies, use_container_width=True)

        fig = px.scatter(df, y=st.session_state['anom_column'], template="plotly_dark",
                        color_discrete_sequence=['#38bdf8'])
        fig.add_scatter(y=anomalies[st.session_state['anom_column']], mode='markers',
                       marker=dict(color='#ef4444', size=12, symbol='x'),
                       name='Anomalies')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No anomalies detected in the selected column")

    st.markdown('</div>', unsafe_allow_html=True)

if 'insight' in st.session_state:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown(f"### AI Insight for '{st.session_state['insight_column']}'")
    st.info(st.session_state['insight'])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
st.markdown("### Complete Statistical Summary")
st.dataframe(df[numeric_cols].describe(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
