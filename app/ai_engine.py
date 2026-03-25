import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# A list of the 3 most stable "Router" endpoints in 2026 to avoid 404 errors
MODEL_ENDPOINTS = [
    "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
    "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct",
    "https://router.huggingface.co/hf-inference/models/google/gemma-2-9b-it"
]

def query_ai(prompt, data_context=""):
    """Cycles through multiple stable models if one returns a 404/Offline error."""
    if not API_KEY:
        return "Error: API Key missing. Please set HUGGINGFACE_API_KEY in your environment."

    full_prompt = f"<s>[INST] Context: {data_context}\n\nQuestion: {prompt} [/INST]"
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 500, 
            "temperature": 0.7, 
            "return_full_text": False
        }
    }

    # Attempt to find a working model
    for url in MODEL_ENDPOINTS:
        try:
            response = requests.post(url, headers=HEADERS, json=payload, timeout=15)
            
            # If successful, return the text immediately
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                return result.get("generated_text", "").strip()
            
            # If 404 or 503, the loop continues to the next model in MODEL_ENDPOINTS
            print(f"Model at {url} returned status {response.status_code}. Trying next...")
            
        except Exception:
            # If connection fails, move to the next model
            continue

    return "AI Error: All model endpoints in the Intellectual Data Lab are currently unreachable. Please check your internet or API key."

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
    """Creates a branded PDF report for Intellectual Data Lab."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header Branding
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(56, 189, 248) 
    pdf.cell(0, 15, "Intellectual Data Lab - Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Section 1: AI Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. AI Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, str(insights))
    
    # Section 2: Findings
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Statistical Findings", ln=True)
    pdf.set_font("Arial", size=11)
    content = f"Predictions:\n{preds}\n\nAnomalies Detected:\n{anomalies}"
    pdf.multi_cell(0, 8, content)
    
    pdf.output(filename)
