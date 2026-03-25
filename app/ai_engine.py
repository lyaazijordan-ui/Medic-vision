import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# UPDATED 2026 ROUTER URL
API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query_ai(prompt, data_context=""):
    """Sends a prompt to the AI using the new Hugging Face Router API."""
    if not API_KEY:
        return "Error: API Key missing. Please set HUGGINGFACE_API_KEY."

    # Combining context with the user question for 'Data Awareness'
    full_prompt = f"<s>[INST] Context: {data_context}\n\nQuestion: {prompt} [/INST]"
    
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        
        if response.status_code == 401:
            return "Error 401: Invalid API Key. Check your Hugging Face token."
        elif response.status_code == 503:
            return "Model is warming up on the router... Try again in a moment."
        elif response.status_code != 200:
            return f"Router Error ({response.status_code}): {response.text}"

        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return result.get("generated_text", "").strip()

    except Exception as e:
        return f"Connection Error: {str(e)}"

def generate_insight(df, column):
    summary = f"Column '{column}' - Mean: {df[column].mean():.2f}, Max: {df[column].max()}, Min: {df[column].min()}."
    return query_ai(f"Provide a professional insight for this data: {summary}")

def medical_analysis(df):
    cols = list(df.columns)
    prompt = f"As a medical analyst, identify risks in these data headers: {cols}. Disclaimer: Not medical advice."
    return query_ai(prompt)

def generate_pdf(filename, insights, preds, anomalies):
    """Creates the PDF with the new branding."""
    pdf = FPDF()
    pdf.add_page()
    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 15, "Intellectual Data Lab - Report", ln=True, align='C')
    pdf.ln(10)
    
    # Body
    pdf.set_font("Arial", size=11)
    content = f"Insights:\n{insights}\n\nPredictions:\n{preds}\n\nAnomalies:\n{anomalies}"
    pdf.multi_cell(0, 8, content)
    
    pdf.output(filename)
