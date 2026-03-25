import requests
from fpdf import FPDF

def generate_pdf(filename, insights, preds, anomalies):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, "AI Data Analysis Report")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Insights: {insights}")
    pdf.ln(5)
    
    pdf.multi_cell(0, 10, f"Predictions: {preds}")
    pdf.ln(5)
    
    # Check if anomalies is a DataFrame or string
    anomalies_str = str(anomalies) if not hasattr(anomalies, 'to_string') else anomalies.to_string()
    pdf.multi_cell(0, 10, f"Anomalies Found: {anomalies_str}")
    
    pdf.output(filename)
import os
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HEADERS = {"Authorization": "Bearer HUGGINGFACE_API_KEY"}

def query_ai(prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    
    if response.status_code != 200:
        return "AI service is busy. Try again."

    try:
        return response.json()[0]["generated_text"]
    except:
        return "Error generating response."


def generate_insight(df, column):
    summary = f"""
    Analyze this dataset column: {column}
    Mean: {df[column].mean()}
    Max: {df[column].max()}
    Min: {df[column].min()}

    Give a simple explanation and possible implications.
    """

    return query_ai(summary)


def medical_analysis(df):
    prompt = f"""
    You are a medical data analyst.

    Dataset columns: {list(df.columns)}

    Identify any abnormal trends or possible health risks.
    Keep it simple and clear. Add disclaimer: Not medical advice.
    """

    return query_ai(prompt)
    prompt = f"""
You are an expert data scientist and medical analyst.

Explain clearly:
- trends
- risks
- recommendations

Be professional but simple.
"""
