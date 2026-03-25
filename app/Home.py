import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Intellectual Data Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    .sidebar-content {
        padding: 1.5rem 1rem;
    }

    .logo-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 2rem;
    }

    .logo-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .logo-text {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .logo-subtitle {
        font-size: 0.7rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .nav-section {
        margin: 2rem 0;
    }

    .nav-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        background: rgba(56, 189, 248, 0.05);
        color: #e2e8f0;
        border: 1px solid rgba(56, 189, 248, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: left;
        margin-bottom: 0.5rem;
    }

    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateX(5px);
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);
    }

    .hero-section {
        text-align: center;
        padding: 4rem 2rem;
        max-width: 900px;
        margin: 0 auto;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        color: #94a3b8;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin-top: 4rem;
        padding: 0 2rem;
    }

    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 10px 40px rgba(56, 189, 248, 0.15);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    .upload-section {
        max-width: 600px;
        margin: 3rem auto;
        padding: 2rem;
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(56, 189, 248, 0.3);
        border-radius: 20px;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-container">
    <div class="logo-icon">🧠</div>
    <div class="logo-text">INTELLECTUAL DATA LAB</div>
    <div class="logo-subtitle">Advanced Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Welcome to the Future of Data Analysis</h1>
    <p class="hero-subtitle">
        Harness the power of AI-driven insights, advanced visualizations, and predictive analytics
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Dashboard Overview</div>
        <div class="feature-description">
            Get instant insights with interactive visualizations and key metrics at a glance
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">Detailed Analytics</div>
        <div class="feature-description">
            10+ advanced chart types including animated scatter plots and regression analysis
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">AI Data Analyst</div>
        <div class="feature-description">
            Ask questions in natural language and get intelligent insights powered by AI
        </div>
    </div>

    <div class="feature-card">
        <div class="feature-icon">📉</div>
        <div class="feature-title">Statistical Summary</div>
        <div class="feature-description">
            Comprehensive statistical analysis with predictions and anomaly detection
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### Get Started")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    import pandas as pd

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.session_state['uploaded_data'] = df
    st.session_state['file_name'] = uploaded_file.name

    st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns")
    st.info("Navigate to any page from the sidebar to start your analysis")

st.markdown('</div>', unsafe_allow_html=True)

if 'uploaded_data' not in st.session_state:
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem; color: #64748b;'>
        <p>Upload a dataset above to unlock all features</p>
    </div>
    """, unsafe_allow_html=True)
