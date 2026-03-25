import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# We use a stable v0.2 URL and a Llama-3 Fallback to avoid 404 errors
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct"

def query_ai(prompt, data_context=""):
    """Sends a prompt to the AI with an automatic fallback if a model is 404 (Offline)."""
    if not API_KEY:
        return "Error: API Key missing. Please set HUGGINGFACE_API_KEY."

    full_prompt = f"<s>[INST] Context: {data_context}\n\nQuestion: {prompt} [/INST]"
    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7, "return_full_text": False}
    }

    try:
        # Attempt Primary Model
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        
        # If 404 or 503, try Fallback
        if response.status_code in [404, 503, 500]:
            response = requests.post(FALLBACK_URL, headers=HEADERS, json=payload, timeout=20)

        if response.status_code != 200:
            return f"AI Router Error ({response.status_code}). The models are currently unavailable."

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return result.get("generated_text", "").strip()

    except Exception as e:
        return f"Connection Error: {str(e)}"

def generate_insight(df, column):
    summary = f"Column '{column}' - Mean: {df[column].mean():.2f}, Max: {df[column].max()}."
    return query_ai(f"Give a short expert insight on these stats: {summary}")

def medical_analysis(df):
    cols = list(df.columns)
    prompt = f"Analyze these medical data headers for risks: {cols}. Disclaimer: Not medical advice."
    return query_ai(prompt)

def generate_pdf(filename, insights, preds, anomalies):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Intellectual Data Lab Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    content = f"Insights: {insights}\n\nPredictions: {preds}\n\nAnomalies: {anomalies}"
    pdf.multi_cell(0, 8, content)
    pdf.output(filename)
