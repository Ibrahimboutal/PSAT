import streamlit as st


def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(120deg, #4f8ef7, #a259ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-sub {
            color: #888;
            font-size: 1.05rem;
            margin-bottom: 1.6rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2d2d4e;
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            text-align: center;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #4f8ef7;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #aaa;
            margin-top: 0.3rem;
        }
        div[data-testid="stSidebar"] {
            background: #0f0f1a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner() -> None:
    st.markdown('<div class="hero-title">💨 PSAT Aerosol Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">3D Monte Carlo · Euler-Maruyama SDE · Numba JIT · Y-bifurcating airway physics</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(col, value: float, label: str, color: str = "#4f8ef7", is_time: bool = False) -> None:
    display_value = f"{value:.2f}s" if is_time else f"{value:.1%}"
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{display_value}</div>
            <div class="metric-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )
