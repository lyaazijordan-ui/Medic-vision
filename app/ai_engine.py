import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free"
]

def query_ai(prompt, data_context=""):
    """Query OpenRouter API with intelligent fallback."""
    if not API_KEY:
        return "Error: API Key missing. Please set OPENROUTER_API_KEY in your environment."

    full_prompt = f"Context: {data_context}\n\nQuestion: {prompt}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://intellectual-data-lab.app",
        "X-Title": "Intellectual Data Lab"
    }

    for model in FALLBACK_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert data analyst providing insights on datasets."},
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }

            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            print(f"Model {model} returned status {response.status_code}. Trying next...")

        except Exception as e:
            print(f"Error with {model}: {e}")
            continue

    return "AI Error: All model endpoints are currently unreachable. Please check your internet or API key."

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
