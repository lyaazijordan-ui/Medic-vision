import requests
import os
from fpdf import FPDF

# 1. SETUP & CONFIGURATION
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
# Using a modern, stable model for 2026
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query_ai(prompt, data_context=""):
    """Sends a prompt to the AI. Can optionally include data context."""
    if not API_KEY:
        return "Error: API Key missing. Please set HUGGINGFACE_API_KEY in your environment."

    # We combine the context of the data with the user's question
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
            return "AI is loading... Please wait a few seconds and try again."
        elif response.status_code != 200:
            return f"AI Error ({response.status_code}): {response.text}"

        result = response.json()
        
        # Handle different response formats from Hugging Face
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No text generated.").strip()
        return result.get("generated_text", "No text generated.").strip()

    except Exception as e:
        return f"Connection Error: {str(e)}"

def generate_insight(df, column):
    """Generates an automated insight based on column statistics."""
    summary = (
        f"The dataset column '{column}' has a mean of {df[column].mean():.2f}, "
        f"a maximum of {df[column].max()}, and a minimum of {df[column].min()}."
    )
    prompt = f"Based on these stats, give a 2-sentence expert observation: {summary}"
    return query_ai(prompt)

def medical_analysis(df):
    """Performs a specialized medical-style data check."""
    # We send the column names and basic stats to give the AI context
    columns = list(df.columns)
    stats = df.describe().to_string()
    
    prompt = f"""
    Acting as a medical data scientist, analyze these data headers: {columns}.
    Summary stats: {stats}
    
    Identify any potential health-related trends or risks. 
    Keep it professional and concise. 
    Disclaimer: This is not medical advice.
    """
    return query_ai(prompt)

def generate_pdf(filename, insights, preds, anomalies):
    """Creates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(56, 189, 248) # Elai Blue
    pdf.cell(0, 15, "Elai AI Data Lab - Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Section: Insights
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. AI Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, str(insights))
    pdf.ln(5)
    
    # Section: Predictions
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Future Predictions", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, str(preds))
    pdf.ln(5)
    
    # Section: Anomalies
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Anomaly Detection", ln=True)
    pdf.set_font("Arial", size=11)
    anomalies_text = anomalies.to_string() if hasattr(anomalies, 'to_string') else str(anomalies)
    pdf.multi_cell(0, 8, anomalies_text)
    
    pdf.output(filename)
