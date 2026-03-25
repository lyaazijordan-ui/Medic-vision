import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as p9
import matplotlib.pyplot as plt
import os

# Import custom modules
from data_engine import predict_future, detect_anomalies
from ai_engine import generate_insight, medical_analysis, query_ai, generate_pdf

# 1. PAGE CONFIG (MUST BE FIRST)
st.set_page_config(page_title="Intellectual Data Lab", layout="wide")

# 2. CUSTOM CSS FOR DARK MODE & CARDS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
    }
    h1, h2, h3 { color: #38bdf8; font-weight: 700; }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white !important;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.title("🧠 Intellectual Data Lab")
st.caption("Precision Analytics | AI-Driven Insights | Statistical Visualization")

# --------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df, use_container_width=True)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.warning("Please upload a dataset with numeric values for full analysis.")
    else:
        # --------- VISUALIZATION SECTION ----------
        st.markdown("## 📈 Interactive Visuals (Plotly)")
        col_x, col_y, col_type = st.columns(3)

        with col_x:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col_y:
            y_axis = st.selectbox("Select Y-axis", numeric_cols)
        with col_type:
            graph_type = st.selectbox("Choose Graph Type", ["Line", "Bar", "Scatter", "Box", "Area"])

        # Plotly Render
        if graph_type == "Line": fig = px.line(df, x=x_axis, y=y_axis)
        elif graph_type == "Bar": fig = px.bar(df, x=x_axis, y=y_axis)
        elif graph_type == "Scatter": fig = px.scatter(df, x=x_axis, y=y_axis)
        elif graph_type == "Box": fig = px.box(df, x=x_axis, y=y_axis)
        else: fig = px.area(df, x=x_axis, y=y_axis)
        
        st.plotly_chart(fig, use_container_width=True)

        # --------- PLOTNINE SECTION (STATISTICAL COLORS) ----------
        st.markdown("## 🎨 Statistical Distribution (Plotnine)")
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p2:
            st.write("### Plot Settings")
            color_theme = st.selectbox("Color Palette", ["viridis", "magma", "inferno", "plasma", "cividis"])
            plot_btn = st.button("Generate Heatmap Distribution")

        with col_p1:
            if plot_btn:
                # Plotnine Logic
                p = (
                    p9.ggplot(df, p9.aes(x=x_axis, y=y_axis, color=y_axis))
                    + p9.geom_point(alpha=0.7, size=3)
                    + p9.theme_minimal()
                    + p9.scale_color_cmap(cmap_name=color_theme)
                    + p9.labs(title=f"Distribution of {y_axis} vs {x_axis}")
                    + p9.theme(figure_size=(8, 4))
                )
                st.pyplot(p9.ggplot.draw(p))

        # --------- ACTION TILES ----------
        st.markdown("## ⚙️ Intelligence Suite")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if st.button("🔮 Predict"):
                st.session_state['preds'] = predict_future(df, y_axis)

        with btn_col2:
            if st.button("🚨 Anomalies"):
                st.session_state['anomalies'] = detect_anomalies(df, y_axis)

        with btn_col3:
            if st.button("🧬 Medical"):
                st.session_state['med'] = medical_analysis(df)

        with btn_col4:
            if st.button("📄 Export PDF"):
                ins = generate_insight(df, y_axis)
                pre = predict_future(df, y_axis)
                ano = detect_anomalies(df, y_axis)
                generate_pdf("Intellectual_Analysis.pdf", ins, pre, ano)
                with open("Intellectual_Analysis.pdf", "rb") as f:
                    st.download_button("⬇️ Download", f, "Intellectual_Analysis.pdf")

        # Display results from session state
        if 'preds' in st.session_state:
            st.write("### 🔮 Predictions", st.session_state['preds'])
        if 'anomalies' in st.session_state:
            st.write("### 🚨 Anomalies", st.session_state['anomalies'])
        if 'med' in st.session_state:
            st.info(st.session_state['med'])

    # --------- SUMMARY CARD ----------
    st.markdown(f"""
    <div class="card">
    <strong>Dataset Intelligence Summary</strong><br>
    Rows: {df.shape[0]} | Columns: {df.shape[1]}<br>
    Primary Metric: {y_axis if numeric_cols else 'N/A'}<br>
    Status: Data processed successfully.
    </div>
    """, unsafe_allow_html=True)

# --------- CHAT SECTION ----------
st.markdown("---")
st.markdown("## 💬 Chat with Intellectual AI")
user_query = st.text_input("Ask a question about your data (e.g., 'What is the trend here?')")

if user_query:
    if uploaded_file is not None:
        # Build context so AI "sees" your data
        ctx = f"Data has {df.shape[0]} rows. Columns: {list(df.columns)}. Focus column: {y_axis if numeric_cols else 'None'}."
        with st.spinner("AI is analyzing..."):
            response = query_ai(user_query, data_context=ctx)
            st.chat_message("assistant").write(response)
    else:
        st.error("Please upload a dataset first.")
