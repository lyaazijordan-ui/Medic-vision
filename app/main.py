from app.data_engine import predict_future, detect_anomalies
from app.ai_engine import generate_insight, medical_analysis, query_ai
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Elai AI Data Lab", layout="wide")

# --------- CUSTOM CSS (THE MAGIC) ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
}
h1, h2, h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# --------- HEADER ----------
st.title("🧠 Elai AI Data Lab")
st.caption("Upload. Visualize. Understand.")

# --------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### 📊 Data Preview")
    st.dataframe(df)

    numeric_cols = df.select_dtypes(include=['number']).columns

    # --------- GRAPH SECTION ----------
    st.markdown("## 📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        x = st.selectbox("Select X-axis", df.columns)
    with col2:
        y = st.selectbox("Select Y-axis", numeric_cols)

    # 10 Graph Types
    graph_type = st.selectbox("Choose Graph Type", [
        "Line", "Bar", "Scatter", "Histogram",
        "Box", "Area", "Pie", "Density Heatmap",
        "Violin", "Strip"
    ])

    if graph_type == "Line":
        fig = px.line(df, x=x, y=y)
    elif graph_type == "Bar":
        fig = px.bar(df, x=x, y=y)
    elif graph_type == "Scatter":
        fig = px.scatter(df, x=x, y=y)
    elif graph_type == "Histogram":
        fig = px.histogram(df, x=x)
    elif graph_type == "Box":
        fig = px.box(df, x=x, y=y)
    elif graph_type == "Area":
        fig = px.area(df, x=x, y=y)
    elif graph_type == "Pie":
        fig = px.pie(df, names=x, values=y)
    elif graph_type == "Density Heatmap":
        fig = px.density_heatmap(df, x=x, y=y)
    elif graph_type == "Violin":
        fig = px.violin(df, x=x, y=y)
    elif graph_type == "Strip":
        fig = px.strip(df, x=x, y=y)

    st.plotly_chart(fig, use_container_width=True)
   
    if st.button("🔮 Predict Future"):
    preds = predict_future(df, y)

    st.write("### 📈 Future Predictions")
    st.write(preds)

    import plotly.graph_objects as go

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(y=df[y], mode='lines', name='Actual'))
    fig2.add_trace(go.Scatter(
        x=list(range(len(df), len(df)+len(preds))),
        y=preds,
        mode='lines',
        name='Predicted'
    ))

    st.plotly_chart(fig2, use_container_width=True)
    if st.button("🧠 Generate AI Insight"):
    insight = generate_insight(df, y)
    st.success(insight)
    if st.button("🧬 Run Medical Analysis"):
    result = medical_analysis(df)
    st.warning(result)
    if st.button("🚨 Detect Anomalies"):
    anomalies = detect_anomalies(df, y)
    st.warning("⚠️ Unusual pattern detected. This may indicate risk in the dataset.")

    if anomalies.empty:
        st.success("No anomalies detected ✅")
    else:
        st.error("Anomalies detected!")
        st.dataframe(anomalies)

    # --------- INSIGHTS ----------
    st.markdown("## 🤖 AI Insights")
    st.markdown(f"""
    <div class="card">
    • Dataset has {df.shape[0]} rows and {df.shape[1]} columns<br>
    • Average of {y}: {df[y].mean():.2f}<br>
    • Max value: {df[y].max()}<br>
    • Trend: {'Increasing' if df[y].iloc[-1] > df[y].iloc[0] else 'Decreasing'}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("## 🤖 Chat with your Data")

user_input = st.text_input("Ask something about your data")

if user_input:
    response = query_ai(user_input)
    st.write(response)
