import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_engine import query_ai, medical_analysis

st.set_page_config(page_title="AI Data Analyst", page_icon="💬", layout="wide")

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

    .chat-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        min-height: 400px;
    }

    .message-user {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.3);
    }

    .message-assistant {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        color: #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .quick-action-btn {
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
    }

    .quick-action-btn:hover {
        background: rgba(56, 189, 248, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
    }

    h1, h2, h3 { color: #38bdf8; }

    .stChatMessage {
        background: transparent !important;
    }

    .info-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(129, 140, 248, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 AI Data Analyst")
st.caption("Ask questions in natural language and get intelligent insights")

if 'uploaded_data' not in st.session_state:
    st.warning("Please upload a dataset on the Home page first")
    st.stop()

df = st.session_state['uploaded_data']
file_name = st.session_state.get('file_name', 'your dataset')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.markdown("---")

st.markdown(f"""
<div class="info-card">
    <strong>Dataset Loaded:</strong> {file_name} <br>
    <strong>Rows:</strong> {df.shape[0]:,} | <strong>Columns:</strong> {df.shape[1]} | <strong>Numeric Fields:</strong> {len(df.select_dtypes(include=['number']).columns)}
</div>
""", unsafe_allow_html=True)

st.markdown("### Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Summarize Data", use_container_width=True):
        st.session_state['quick_question'] = "Give me a brief summary of this dataset"

with col2:
    if st.button("Find Correlations", use_container_width=True):
        st.session_state['quick_question'] = "What are the key correlations in this data?"

with col3:
    if st.button("Identify Trends", use_container_width=True):
        st.session_state['quick_question'] = "What trends can you identify in this data?"

with col4:
    if st.button("Medical Analysis", use_container_width=True):
        with st.spinner("Running medical analysis..."):
            result = medical_analysis(df)
            st.session_state['chat_history'].append({
                'role': 'user',
                'content': 'Perform a medical analysis on this dataset'
            })
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'content': result
            })

st.markdown("---")

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state['chat_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])

st.markdown('</div>', unsafe_allow_html=True)

user_question = st.chat_input("Ask a question about your data...")

if 'quick_question' in st.session_state:
    user_question = st.session_state['quick_question']
    del st.session_state['quick_question']

if user_question:
    st.session_state['chat_history'].append({
        'role': 'user',
        'content': user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    context = f"""
    Dataset: {file_name}
    Total Rows: {df.shape[0]}
    Total Columns: {df.shape[1]}
    Column Names: {list(df.columns)}
    Numeric Columns: {numeric_cols}
    Sample Data (first 3 rows): {df.head(3).to_dict('records')}
    """

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = query_ai(user_question, context)
            st.write(response)

    st.session_state['chat_history'].append({
        'role': 'assistant',
        'content': response
    })

    st.rerun()

if len(st.session_state['chat_history']) > 0:
    if st.button("Clear Chat History"):
        st.session_state['chat_history'] = []
        st.rerun()
