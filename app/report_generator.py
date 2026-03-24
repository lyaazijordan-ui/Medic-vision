from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(filename, insights, predictions, anomalies):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Data Analysis Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Insights:", styles["Heading2"]))
    content.append(Paragraph(str(insights), styles["BodyText"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Predictions:", styles["Heading2"]))
    content.append(Paragraph(str(predictions), styles["BodyText"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("Anomalies:", styles["Heading2"]))
    content.append(Paragraph(str(anomalies), styles["BodyText"]))

    doc.build(content)
