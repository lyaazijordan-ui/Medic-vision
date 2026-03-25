import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as p9
import matplotlib.pyplot as plt
import os

# Local Imports
from data_engine import predict_future, detect_anomalies
from ai_engine import generate_insight, medical_analysis, query_ai, generate_pdf

# 1. CONFIGURATION
st.set_page_config(page_title="Intellectual Data Lab", layout="wide")

# 2. CUSTOM THEMING
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: white; }
    h1, h2, h3 { color: #38bdf8; font-weight: 700; }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%; border-radius: 12px;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white !important; font-weight: bold; border: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Intellectual Data Lab")
st.caption("Strategic Intelligence & Advanced Visualization Suite")

# --------- DATA INGESTION ----------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.warning("Numeric columns required for advanced plotting.")
    else:
        # --------- 10-GRAPH ANALYTICS SUITE ----------
        st.markdown("## 📈 The 10-Graph Analytics Suite")
        
        col_ctrl, col_view = st.columns([1, 3])

        with col_ctrl:
            st.write("### Graph Controls")
            x_axis = st.selectbox("X-Axis", df.columns)
            y_axis = st.selectbox("Y-Axis", numeric_cols)
            graph_choice = st.selectbox("Visual Type", [
                "1. Interactive Line", "2. Animated Scatter", "3. Multi-Bar", 
                "4. Statistical Boxplot", "5. Area Chart", "6. Heatmap Density", 
                "7. Donut Composition", "8. Violin Distribution", 
                "9. Plotnine Regression", "10. Plotnine Facet Grid"
            ])
            color_theme = st.selectbox("Plotnine Palette", ["viridis", "magma", "plasma", "inferno"])

        with col_view:
            if "1." in graph_choice: fig = px.line(df, x=x_axis, y=y_axis, template="plotly_dark")
            elif "2." in graph_choice: fig = px.scatter(df, x=x_axis, y=y_axis, color=y_axis, size=y_axis, template="plotly_dark")
            elif "3." in graph_choice: fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, template="plotly_dark")
            elif "4." in graph_choice: fig = px.box(df, x=x_axis, y=y_axis, points="all", template="plotly_dark")
            elif "5." in graph_choice: fig = px.area(df, x=x_axis, y=y_axis, template="plotly_dark")
            elif "6." in graph_choice: fig = px.density_heatmap(df, x=x_axis, y=y_axis, text_auto=True, template="plotly_dark")
            elif "7." in graph_choice: fig = px.pie(df, names=x_axis, values=y_axis, hole=0.4, template="plotly_dark")
            elif "8." in graph_choice: fig = px.violin(df, y=y_axis, x=x_axis, box=True, points="all", template="plotly_dark")
            elif "9." in graph_choice:
                p = (p9.ggplot(df, p9.aes(x=x_axis, y=y_axis)) + p9.geom_point(p9.aes(color=y_axis)) 
                     + p9.geom_smooth(method='lm', color='red') + p9.scale_color_cmap(cmap_name=color_theme) + p9.theme_minimal())
                st.pyplot(p9.ggplot.draw(p))
                fig = None
            elif "10." in graph_choice:
                p = (p9.ggplot(df, p9.aes(x=x_axis, y=y_axis)) + p9.geom_col(fill="#38bdf8") 
                     + p9.facet_wrap(f'~{x_axis}', scales='free_y') + p9.theme_dark())
                st.pyplot(p9.ggplot.draw(p))
                fig = None

            if fig: st.plotly_chart(fig, use_container_width=True)

        # --------- ACTION HUB ----------
        st.markdown("## ⚙️ Intelligence Operations")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.button("🔮 Predict"): st.session_state['p'] = predict_future(df, y_axis)
        with c2:
            if st.button("🚨 Anomalies"): st.session_state['a'] = detect_anomalies(df, y_axis)
        with c3:
            if st.button("🧬 Medical"): st.session_state['m'] = medical_analysis(df)
        with c4:
            if st.button("📄 Export Report"):
                generate_pdf("Intellectual_Analysis.pdf", "AI Insight", "Predictions", "Anomalies")
                with open("Intellectual_Analysis.pdf", "rb") as f:
                    st.download_button("⬇️ Download PDF", f, "Intellectual_Analysis.pdf")

        if 'p' in st.session_state: st.write("### Predictions", st.session_state['p'])
        if 'm' in st.session_state: st.info(st.session_state['m'])

# --------- CHAT SECTION ----------
st.markdown("---")
st.markdown("## 💬 Chat with Intellectual AI")
q = st.text_input("Ask about your data...")
if q and uploaded_file:
    ctx = f"Rows: {df.shape[0]}. Columns: {list(df.columns)}. Target: {y_axis}."
    with st.spinner("Analyzing..."):
        st.chat_message("assistant").write(query_ai(q, ctx))
