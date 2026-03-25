# Intellectual Data Lab

A modern, multi-page data analytics platform powered by AI, featuring advanced visualizations, predictive analytics, and intelligent insights.

## Features

### 1. Dashboard Overview
- Real-time metrics and KPIs
- Interactive data previews
- Correlation analysis
- Distribution visualizations

### 2. Detailed Analytics
- 10+ advanced chart types
- Interactive visualizations with Plotly
- Statistical plots with Plotnine
- Customizable themes and color palettes

### 3. AI Data Analyst
- Natural language query interface
- Context-aware insights powered by OpenRouter
- Medical data analysis capabilities
- Interactive chat history

### 4. Statistical Summary
- Predictive modeling with linear regression
- Anomaly detection using statistical methods
- AI-generated insights
- PDF report generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Add your OpenRouter API key to `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from: https://openrouter.ai/

## Running the Application

```bash
streamlit run app/Home.py
```

The application will open in your browser with a modern glassmorphism sidebar for easy navigation.

## Architecture

The application uses Streamlit's multi-page architecture:

- `Home.py` - Landing page and data upload
- `pages/1_Dashboard_Overview.py` - Main dashboard
- `pages/2_Detailed_Analytics.py` - Advanced visualizations
- `pages/3_AI_Data_Analyst.py` - AI chat interface
- `pages/4_Statistical_Summary.py` - Statistical operations

Data uploaded on the Home page persists across all pages using Streamlit's session state.

## Technologies

- **Frontend**: Streamlit with custom CSS (glassmorphism design)
- **Visualizations**: Plotly, Plotnine, Matplotlib
- **AI**: OpenRouter API (with fallback models)
- **Analytics**: Pandas, NumPy, Scikit-learn
- **Reports**: FPDF

## Design

The UI features a modern dark theme with:
- Glassmorphism effects
- Smooth transitions and hover states
- Gradient accents (blue/purple palette)
- Responsive layouts
- Interactive components
