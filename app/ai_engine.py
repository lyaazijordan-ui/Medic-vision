import requests
from fpdf import FPDF
import os

# 1. Correctly fetch the API Key
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
# 2. FIXED: Use an f-string to pass the actual variable, not the string "HUGGINGFACE_API_KEY"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def generate_pdf(filename, insights, preds, anomalies):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, "AI Data Analysis Report")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Insights: {insights}")
    pdf.ln(5)
    
    # Simple check for prediction format
    pdf.multi_cell(0, 10, f"Predictions: {preds}")
    pdf.ln(5)
    
    anomalies_str = str(anomalies) if not hasattr(anomalies, 'to_string') else anomalies.to_string()
    pdf.multi_cell(0, 10, f"Anomalies Found: {anomalies_str}")
    
    pdf.output(filename)

def query_ai(prompt):
    # Added a check to ensure API_KEY exists
    if not API_KEY:
        return "Error: HUGGINGFACE_API_KEY not found in environment variables."

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    
    if response.status_code != 200:
        return f"AI service error ({response.status_code}). Try again later."

    try:
        return response.json()[0]["generated_text"]
    except (KeyError, IndexError):
        return "Error parsing AI response."

def generate_insight(df, column):
    summary = f"""
    Analyze this dataset column: {column}
    Mean: {df[column].mean():.2f}
    Max: {df[column].max()}
    Min: {df[column].min()}

    Give a simple explanation and possible implications.
    """
    return query_ai(summary)

def medical_analysis(df):
    # 3. FIXED: Cleaned up the prompt and removed unreachable code
    prompt = f"""
    You are a medical data analyst.
    Dataset columns: {list(df.columns)}

    Identify any abnormal trends or possible health risks based on these headers.
    Keep it simple and clear. 
    Include: trends, risks, and general recommendations.
    
    Disclaimer: Not medical advice.
    """
    return query_ai(prompt)
