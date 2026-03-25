import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 2026 Router Endpoints (Stable Aliases)
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_URL = "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta"

def query_ai(prompt, data_context=""):
    """Sends a prompt to the AI with an automatic fallback if a model is Offline."""
    if not API_KEY:
        return "Error: API Key missing. Please set HUGGINGFACE_API_KEY in your environment."

    full_prompt = f"<s>[INST] Context: {data_context}\n\nQuestion: {prompt} [/INST]"
    payload = {
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        
        # Fallback logic for 404/503/500 errors
        if response.status_code in [404, 503, 500]:
            response = requests.post(FALLBACK_URL, headers=HEADERS, json=payload, timeout=20)

        if response.status_code != 200:
            return f"AI Router Error ({response.status_code}). Please try again in a few seconds."

        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return result.get("generated_text", "").strip()

    except Exception as e:
        return f"Connection Error: {str(e)}"

def generate_insight(df, column):
    """Generates an automated summary insight."""
    summary = f"Column '{column}' - Mean: {df[column].mean():.2f}, Max: {df[column].max()}."
    return query_ai(f"Give a short expert insight on these stats: {summary}")

def medical_analysis(df):
    """Specialized medical data check."""
    cols = list(df.columns)
    prompt = f"Identify potential health-related risks or trends in these data headers: {cols}. Disclaimer: Not medical advice."
    return query_ai(prompt)

def generate_pdf(filename, insights, preds, anomalies):
    """Creates a branded PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(56, 189, 248) 
    pdf.cell(0, 15, "Intellectual Data Lab - Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. AI Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, str(insights))
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Statistical Findings", ln=True)
    pdf.set_font("Arial", size=11)
    content = f"Predictions: {preds}\n\nAnomalies: {anomalies}"
    pdf.multi_cell(0, 8, content)
    
    pdf.output(filename)
