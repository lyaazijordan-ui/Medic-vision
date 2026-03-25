import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotnine as p9
import matplotlib.pyplot as plt

st.set_page_config(page_title="Detailed Analytics", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    .control-panel {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .chart-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    h1, h2, h3 { color: #38bdf8; }

    .stSelectbox label, .stRadio label {
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Detailed Analytics")
st.caption("10+ Advanced Visualization Types")

if 'uploaded_data' not in st.session_state:
    st.warning("Please upload a dataset on the Home page first")
    st.stop()

df = st.session_state['uploaded_data']
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in the dataset")
    st.stop()

st.markdown("---")

col_ctrl, col_view = st.columns([1, 3])

with col_ctrl:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### Chart Controls")

    x_axis = st.selectbox("X-Axis", df.columns, key="x_axis")
    y_axis = st.selectbox("Y-Axis", numeric_cols, key="y_axis")

    chart_types = [
        "Interactive Line",
        "Animated Scatter",
        "Multi-Bar Chart",
        "Statistical Boxplot",
        "Area Chart",
        "Density Heatmap",
        "Donut Composition",
        "Violin Distribution",
        "Plotnine Regression",
        "Plotnine Facet Grid"
    ]

    graph_choice = st.selectbox("Visualization Type", chart_types, key="graph_type")
    color_theme = st.selectbox("Color Theme", ["viridis", "magma", "plasma", "inferno", "Blues", "Turbo"])

    st.markdown('</div>', unsafe_allow_html=True)

with col_view:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown(f"### {graph_choice}")

    try:
        if graph_choice == "Interactive Line":
            fig = px.line(df, x=x_axis, y=y_axis, template="plotly_dark",
                         color_discrete_sequence=['#38bdf8'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Animated Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=y_axis, size=y_axis,
                           template="plotly_dark", color_continuous_scale=color_theme)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Multi-Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, template="plotly_dark")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Statistical Boxplot":
            fig = px.box(df, x=x_axis, y=y_axis, points="all", template="plotly_dark",
                        color_discrete_sequence=['#38bdf8'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Area Chart":
            fig = px.area(df, x=x_axis, y=y_axis, template="plotly_dark",
                         color_discrete_sequence=['#38bdf8'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Density Heatmap":
            fig = px.density_heatmap(df, x=x_axis, y=y_axis, text_auto=True,
                                    template="plotly_dark", color_continuous_scale=color_theme)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Donut Composition":
            fig = px.pie(df, names=x_axis, values=y_axis, hole=0.4, template="plotly_dark")
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Violin Distribution":
            fig = px.violin(df, y=y_axis, x=x_axis, box=True, points="all",
                          template="plotly_dark", color_discrete_sequence=['#38bdf8'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)

        elif graph_choice == "Plotnine Regression":
            p = (p9.ggplot(df, p9.aes(x=x_axis, y=y_axis))
                 + p9.geom_point(p9.aes(color=y_axis), size=3, alpha=0.7)
                 + p9.geom_smooth(method='lm', color='#38bdf8', size=1.5)
                 + p9.scale_color_cmap(cmap_name=color_theme)
                 + p9.theme_minimal()
                 + p9.theme(
                     figure_size=(10, 6),
                     panel_background=p9.element_rect(fill='#1e293b'),
                     plot_background=p9.element_rect(fill='#1e293b'),
                     text=p9.element_text(color='#e2e8f0')
                 ))
            st.pyplot(p9.ggplot.draw(p))

        elif graph_choice == "Plotnine Facet Grid":
            p = (p9.ggplot(df, p9.aes(x=x_axis, y=y_axis))
                 + p9.geom_col(fill="#38bdf8", alpha=0.8)
                 + p9.facet_wrap(f'~{x_axis}', scales='free_y')
                 + p9.theme_minimal()
                 + p9.theme(
                     figure_size=(12, 8),
                     panel_background=p9.element_rect(fill='#1e293b'),
                     plot_background=p9.element_rect(fill='#1e293b'),
                     text=p9.element_text(color='#e2e8f0')
                 ))
            st.pyplot(p9.ggplot.draw(p))

    except Exception as e:
        st.error(f"Could not generate chart: {str(e)}")
        st.info("Try selecting different columns or chart types")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.info("Hover over charts for interactive tooltips and data exploration")
