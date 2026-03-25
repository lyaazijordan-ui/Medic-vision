import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Overview", page_icon="📊", layout="wide")

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

    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 48px rgba(56, 189, 248, 0.2);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    .chart-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    h1, h2, h3 { color: #38bdf8; }

    .stDataFrame {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Dashboard Overview")
st.caption("Real-time insights and key metrics")

if 'uploaded_data' not in st.session_state:
    st.warning("Please upload a dataset on the Home page first")
    st.stop()

df = st.session_state['uploaded_data']

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[0]:,}</div>
        <div class="metric-label">Total Rows</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]}</div>
        <div class="metric-label">Total Columns</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    numeric_cols = df.select_dtypes(include=['number']).columns
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(numeric_cols)}</div>
        <div class="metric-label">Numeric Fields</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    missing = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{missing}</div>
        <div class="metric-label">Missing Values</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### Data Preview")
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if len(numeric_cols) > 0:
    st.markdown("### Quick Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Distribution Overview")
        selected_col = st.selectbox("Select column", numeric_cols, key="dist_col")
        fig = px.histogram(df, x=selected_col, marginal="box", template="plotly_dark",
                          color_discrete_sequence=['#38bdf8'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Correlation Heatmap")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, template="plotly_dark",
                       color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### Statistical Summary")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
