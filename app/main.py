import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Import custom modules
from data_engine import predict_future, detect_anomalies
from ai_engine import generate_insight, medical_analysis, query_ai, generate_pdf

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Elai AI Data Lab", layout="wide")

API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 2. CONSOLIDATED CSS
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
    }
    button {
        border-radius: 12px !important;
        background: linear-gradient(90deg, #38bdf8, #6366f1);
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.title("🧠 Elai AI Data Lab")
st.caption("Upload. Visualize. Understand.")

# --------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Handle different file types
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        st.error("This dataset has no numeric columns for analysis.")
    else:
        # --------- GRAPH SECTION ----------
        st.markdown("## 📈 Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            x = st.selectbox("Select X-axis", df.columns)
        with col2:
            y = st.selectbox("Select Y-axis", numeric_cols)

        graph_type = st.selectbox("Choose Graph Type", [
            "Line", "Bar", "Scatter", "Histogram", "Box", 
            "Area", "Pie", "Density Heatmap", "Violin", "Strip"
        ])

        # Plotly Logic
        if graph_type == "Line": fig = px.line(df, x=x, y=y)
        elif graph_type == "Bar": fig = px.bar(df, x=x, y=y)
        elif graph_type == "Scatter": fig = px.scatter(df, x=x, y=y)
        elif graph_type == "Histogram": fig = px.histogram(df, x=x)
        elif graph_type == "Box": fig = px.box(df, x=x, y=y)
        elif graph_type == "Area": fig = px.area(df, x=x, y=y)
        elif graph_type == "Pie": fig = px.pie(df, names=x, values=y)
        elif graph_type == "Density Heatmap": fig = px.density_heatmap(df, x=x, y=y)
        elif graph_type == "Violin": fig = px.violin(df, x=x, y=y)
        else: fig = px.strip(df, x=x, y=y)

        st.plotly_chart(fig, use_container_width=True)

        # --------- ANALYSIS BUTTONS (Corrected Indentation) ----------
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button("🔮 Predict Future"):
                preds = predict_future(df, y)
                st.write("### 📈 Future Predictions")
                st.write(preds)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=df[y], mode='lines', name='Actual'))
                fig2.add_trace(go.Scatter(
                    x=list(range(len(df), len(df) + len(preds))),
                    y=preds, mode='lines', name='Predicted'
                ))
                st.plotly_chart(fig2, use_container_width=True)

        with col_btn2:
            if st.button("📄 Generate Report"):
                insights = generate_insight(df, y)
                preds = predict_future(df, y)
                anomalies = detect_anomalies(df, y)
                generate_pdf("report.pdf", insights, preds, anomalies)
                
                with open("report.pdf", "rb") as f:
                    st.download_button("⬇️ Download Report", f, "AI_Report.pdf")

        with col_btn3:
            if st.button("🚨 Detect Anomalies"):
                anomalies = detect_anomalies(df, y)
                if anomalies.empty:
                    st.success("No anomalies detected ✅")
                else:
                    st.error("Anomalies detected!")
                    st.dataframe(anomalies)

        # --------- AI INSIGHTS ----------
        st.markdown("## 🤖 AI Insights")
        if st.button("🧠 Generate AI Insight"):
            insight_text = generate_insight(df, y)
            st.markdown(f'<div class="card">{insight_text}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
        • Dataset has {df.shape[0]} rows and {df.shape[1]} columns<br>
        • Average of {y}: {df[y].mean():.2f}<br>
        • Max value: {df[y].max()}<br>
        • Trend: {'Increasing' if df[y].iloc[-1] > df[y].iloc[0] else 'Decreasing'}
        </div>
        """, unsafe_allow_html=True)

# --------- CHAT SECTION ----------
st.markdown("## 💬 Chat with your Data")
user_input = st.text_input("Ask something about your data")
if user_input:
    response = query_ai(user_input)
    st.write(response)
# In main.py
if user_input:
    # Create a small summary of the data so the AI knows what you are talking about
    data_summary = f"The dataset has {df.shape[0]} rows. Columns: {list(df.columns)}. Focus column: {y}"
    
    # Pass both the question AND the summary
    response = query_ai(user_input, data_context=data_summary)
    st.write(response)

