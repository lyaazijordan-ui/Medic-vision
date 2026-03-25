import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import custom modules (Local imports since main.py is in the app folder)
from data_engine import predict_future, detect_anomalies
from ai_engine import generate_insight, medical_analysis, query_ai, generate_pdf

# 1. PAGE CONFIG (MUST BE FIRST)
st.set_page_config(page_title="Intellectual Data Lab", layout="wide")

# 2. CUSTOM CSS
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
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.title("🧠 Intellectual Data Lab")
st.caption("Advanced Analytics & AI-Driven Insights")

# --------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Logic to handle file types
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df, use_container_width=True)

    # Get numeric columns for plotting
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns found for visualization. Please upload a dataset with numbers.")
    else:
        # --------- GRAPH SECTION ----------
        st.markdown("## 📈 Visualizations")
        col_x, col_y, col_type = st.columns([1, 1, 1])

        with col_x:
            x_axis = st.selectbox("Select X-axis", df.columns)
        with col_y:
            y_axis = st.selectbox("Select Y-axis", numeric_cols)
        with col_type:
            graph_type = st.selectbox("Choose Graph Type", [
                "Line", "Bar", "Scatter", "Histogram", "Box", 
                "Area", "Pie", "Violin"
            ])

        # Plotly logic
        if graph_type == "Line": fig = px.line(df, x=x_axis, y=y_axis)
        elif graph_type == "Bar": fig = px.bar(df, x=x_axis, y=y_axis)
        elif graph_type == "Scatter": fig = px.scatter(df, x=x_axis, y=y_axis)
        elif graph_type == "Histogram": fig = px.histogram(df, x=x_axis)
        elif graph_type == "Box": fig = px.box(df, x=x_axis, y=y_axis)
        elif graph_type == "Area": fig = px.area(df, x=x_axis, y=y_axis)
        elif graph_type == "Pie": fig = px.pie(df, names=x_axis, values=y_axis)
        else: fig = px.violin(df, x=x_axis, y=y_axis)

        st.plotly_chart(fig, use_container_width=True)

        # --------- ACTION BUTTONS ----------
        st.markdown("## ⚙️ Data Operations")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        with btn_col1:
            if st.button("🔮 Predict Future"):
                preds = predict_future(df, y_axis)
                st.session_state['preds'] = preds
                st.success("Predictions Generated!")

        with btn_col2:
            if st.button("🚨 Detect Anomalies"):
                anomalies = detect_anomalies(df, y_axis)
                st.session_state['anomalies'] = anomalies
                st.warning("Anomaly Scan Complete.")

        with btn_col3:
            if st.button("🧬 Medical Analysis"):
                med_result = medical_analysis(df)
                st.session_state['med_analysis'] = med_result
                st.info("Medical Insights Ready.")

        with btn_col4:
            if st.button("📄 Generate PDF Report"):
                # Collecting current states for the PDF
                ins = generate_insight(df, y_axis)
                pre = predict_future(df, y_axis)
                ano = detect_anomalies(df, y_axis)
                generate_pdf("Intellectual_Report.pdf", ins, pre, ano)
                with open("Intellectual_Report.pdf", "rb") as f:
                    st.download_button("⬇️ Download Report", f, "Intellectual_Report.pdf")

        # --------- DISPLAY RESULTS ----------
        if 'preds' in st.session_state:
            st.write("### 📈 Future Predictions")
            st.write(st.session_state['preds'])
            
        if 'anomalies' in st.session_state:
            st.write("### ⚠️ Detected Anomalies")
            st.dataframe(st.session_state['anomalies'])

        if 'med_analysis' in st.session_state:
            st.markdown(f'<div class="card"><strong>Medical Analysis:</strong><br>{st.session_state["med_analysis"]}</div>', unsafe_allow_html=True)

    # --------- INSIGHTS SUMMARY CARD ----------
    st.markdown("## 🤖 Dataset Summary")
    st.markdown(f"""
    <div class="card">
    • <strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]}<br>
    • <strong>Analysis Target:</strong> {y_axis if numeric_cols else 'N/A'}<br>
    • <strong>Current Trend:</strong> {'Calculating...' if not numeric_cols else ('Increasing' if df[y_axis].iloc[-1] > df[y_axis].iloc[0] else 'Decreasing')}
    </div>
    """, unsafe_allow_html=True)

# --------- CHAT SECTION ----------
st.markdown("---")
st.markdown("## 💬 Chat with Intellectual AI")
user_query = st.text_input("Ask a specific question about this dataset:")

if user_query:
    if uploaded_file is not None:
        # Context building for the AI
        data_context = f"Dataset has {df.shape[0]} rows. Columns: {list(df.columns)}. Target column: {y_axis if numeric_cols else 'None'}."
        with st.spinner("Analyzing data..."):
            ai_response = query_ai(user_query, data_context=data_context)
            st.chat_message("assistant").write(ai_response)
    else:
        st.error("Please upload a file first so I can assist you with the data!")
