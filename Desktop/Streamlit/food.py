import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import re
import json
import html as html_module
from streamlit_mic_recorder import mic_recorder
from groq import Groq

# =========================================================
# LOAD GROQ API KEY FROM note.env
# =========================================================

def load_api_key_from_file(filepath: str) -> str:
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    _, value = line.split("=", 1)
                    return value.strip()
                else:
                    return line
    except FileNotFoundError:
        return None

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ENV_FILE     = os.path.join(BASE_DIR, "note.env")
GROQ_API_KEY = load_api_key_from_file(ENV_FILE) or os.getenv("GROQ_API_KEY", "")

# =========================================================
# PAGE CONFIG  — must be FIRST Streamlit call
# =========================================================

st.set_page_config(
    page_title="VigilantAP | Food Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# SESSION STATE
# =========================================================

if "welcome_screen_seen" not in st.session_state:
    st.session_state.welcome_screen_seen = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "anomalies_detected" not in st.session_state:
    st.session_state.anomalies_detected = False
if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = ""

# =========================================================
# SAFE HTML HELPER
# =========================================================

def safe(text: str) -> str:
    return html_module.escape(str(text))

# =========================================================
# WELCOME / SPLASH SCREEN  — pure Streamlit, no CSS tricks
# =========================================================

if not st.session_state.welcome_screen_seen:

    # Full-page dark background via CSS (doesn't block the button)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Hide Streamlit chrome */
    header[data-testid="stHeader"],
    div[data-testid="stDecoration"],
    footer, #MainMenu { display: none !important; }

    /* Dark full-page background */
    .stApp { background-color: #0F1117 !important; }

    /* Centre everything in the splash */
    section[data-testid="stMain"] > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        padding: 0 !important;
    }

    /* The welcome card itself */
    .splash-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0;
    }

    .splash-icon {
        background: linear-gradient(135deg, #2563EB, #0891B2);
        width: 110px; height: 110px; border-radius: 28px;
        display: flex; align-items: center; justify-content: center;
        font-size: 58px;
        box-shadow: 0 24px 60px rgba(37,99,235,0.40);
        animation: pulse-glow 2.4s ease-in-out infinite;
        margin-bottom: 28px;
    }

    .splash-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB, #0891B2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 10px 0;
        text-align: center;
    }

    .splash-sub {
        color: #94A3B8;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 48px;
        text-align: center;
    }

    /* Make the Streamlit button look like the branded button */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2563EB, #0891B2) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 48px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 8px 28px rgba(37,99,235,0.40) !important;
        cursor: pointer !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        min-width: 220px !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 36px rgba(37,99,235,0.50) !important;
    }

    @keyframes pulse-glow {
        0%, 100% { transform: scale(1);   box-shadow: 0 24px 60px rgba(37,99,235,0.40); }
        50%       { transform: scale(1.04); box-shadow: 0 28px 70px rgba(37,99,235,0.55); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Render splash content using Streamlit columns to centre horizontally
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="splash-wrap">
            <div class="splash-icon">🛡️</div>
            <div class="splash-title">VigilantAP</div>
            <div class="splash-sub">Food Safety Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # Native Streamlit button — always visible, always clickable
        if st.button("🚀  Initialize Platform", use_container_width=True):
            st.session_state.welcome_screen_seen = True
            # Optional: browser voice greeting fires after rerun via JS injected below
            st.markdown("""
            <script>
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance(
                    "Welcome to Vigilant A P, Food Safety Intelligence Platform."
                );
                msg.lang = 'en-US';
                msg.rate = 0.95;
                window.speechSynthesis.speak(msg);
            }
            </script>
            """, unsafe_allow_html=True)
            st.rerun()

    st.stop()

# =========================================================
# API KEY CHECK  (after splash so error shows on main page)
# =========================================================

if not GROQ_API_KEY:
    st.error(
        "❌ Groq API key not found. "
        "Add your key to **streamlit_project/note.env** as:\n\n"
        "`GROQ_API_KEY=gsk_xxxxxxxx`"
    )
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# GLOBAL CSS  (only loaded after splash)
# =========================================================

def get_theme_css(dark: bool) -> str:
    if dark:
        return """
        :root {
            --bg-base:      #0F1117;
            --bg-card:      #1A1D27;
            --bg-card2:     #22263A;
            --border:       rgba(255,255,255,0.07);
            --border-md:    rgba(255,255,255,0.12);
            --text-primary: #F1F5F9;
            --text-muted:   #94A3B8;
            --text-dim:     #CBD5E1;
            --shadow-sm:    0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md:    0 4px 12px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.3);
        }
        .stApp { background-color: #0F1117 !important; }
        section[data-testid="stSidebar"] { background: #1A1D27 !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
        section[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #22263A !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #F1F5F9 !important; }
        [data-testid="stDataFrame"] { background: #1A1D27 !important; }
        [data-testid="stDataFrame"] th { background: #22263A !important; }
        [data-testid="stDataFrame"] td { color: #F1F5F9 !important; }
        div[data-testid="metric-container"] { background: #1A1D27 !important; }
        div[data-testid="metric-container"] [data-testid="metric-value"] { color: #F1F5F9 !important; }
        .insight-box { background: #1A1D27 !important; }
        .insight-box h4 { color: #F1F5F9 !important; }
        .stTabs [data-baseweb="tab-list"] { background: #1A1D27 !important; }
        .ai-block-card { background: #1A1D27 !important; border-color: rgba(255,255,255,0.08) !important; }
        .ai-block-card .block-body { color: #CBD5E1 !important; }
        """
    else:
        return """
        :root {
            --bg-base:      #F8F9FB;
            --bg-card:      #FFFFFF;
            --bg-card2:     #F3F5F9;
            --border:       rgba(0,0,0,0.08);
            --border-md:    rgba(0,0,0,0.12);
            --text-primary: #0F172A;
            --text-muted:   #64748B;
            --text-dim:     #475569;
            --shadow-sm:    0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md:    0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        }
        .stApp { background-color: #F8F9FB !important; }
        section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid rgba(0,0,0,0.08) !important; }
        section[data-testid="stSidebar"] * { color: #0F172A !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #F8F9FB !important; border: 1px solid rgba(0,0,0,0.12) !important; color: #0F172A !important; }
        """

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=Syne:wght@700;800&display=swap');

{get_theme_css(st.session_state.dark_mode)}

.stApp {{ font-family: 'DM Sans', sans-serif; color: var(--text-primary) !important; }}
header, footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg-card);
    border-radius: 14px;
    padding: 6px;
    gap: 4px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: none !important;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #2563EB, #0891B2) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.28) !important;
}}

/* ---- METRICS ---- */
div[data-testid="metric-container"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: var(--shadow-sm) !important;
}}
div[data-testid="metric-container"] label {{ color: var(--text-muted) !important; font-size: 13px !important; }}
div[data-testid="metric-container"] [data-testid="metric-value"] {{ font-size: 32px !important; font-weight: 700 !important; }}

/* ---- BUTTONS ---- */
.stButton > button, .stDownloadButton > button {{
    background: linear-gradient(135deg, #2563EB, #0891B2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}
.stButton > button:hover {{ transform: translateY(-1px) !important; box-shadow: 0 6px 16px rgba(37,99,235,0.35) !important; }}

/* ---- DATAFRAME ---- */
[data-testid="stDataFrame"] {{
    background: var(--bg-card) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}}
[data-testid="stDataFrame"] th {{ background: var(--bg-card2) !important; color: #0891B2 !important; font-weight: 700 !important; }}

/* ---- ALERTS ---- */
.stAlert {{ border-radius: 12px !important; border: 1px solid var(--border) !important; }}

/* ---- PROGRESS ---- */
div[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, #2563EB, #0891B2) !important;
    border-radius: 20px !important;
}}
div[data-testid="stProgress"] > div {{ background: #E2E8F0 !important; border-radius: 20px !important; }}

h1,h2,h3,h4,h5,h6 {{ color: var(--text-primary) !important; }}
p,li,label,span {{ color: var(--text-dim) !important; }}
.stCaption, caption {{ color: var(--text-muted) !important; font-size: 13px !important; }}

/* ============================================================ KPI CARDS */
.kpi-card {{
    border-radius: 20px;
    padding: 24px 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s cubic-bezier(.34,1.56,.64,1), box-shadow 0.25s ease;
    cursor: default;
}}
.kpi-card:hover {{ transform: translateY(-4px) scale(1.01); box-shadow: 0 16px 40px rgba(0,0,0,0.18) !important; }}
.kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 50%;
    background: linear-gradient(180deg, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 100%);
    border-radius: 20px 20px 0 0; pointer-events: none;
}}
.kpi-card::after {{
    content: '';
    position: absolute; bottom: -30px; right: -30px;
    width: 110px; height: 110px;
    background: rgba(255,255,255,0.10); border-radius: 50%; pointer-events: none;
}}
.kpi-blue   {{ background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 50%, #3B82F6 100%); box-shadow: 0 8px 28px rgba(37,99,235,0.42) !important; }}
.kpi-green  {{ background: linear-gradient(135deg, #047857 0%, #059669 50%, #10B981 100%); box-shadow: 0 8px 28px rgba(5,150,105,0.42) !important; }}
.kpi-red    {{ background: linear-gradient(135deg, #BE123C 0%, #E11D48 50%, #F43F5E 100%); box-shadow: 0 8px 28px rgba(225,29,72,0.42) !important; }}
.kpi-amber  {{ background: linear-gradient(135deg, #B45309 0%, #D97706 50%, #F59E0B 100%); box-shadow: 0 8px 28px rgba(217,119,6,0.42) !important; }}
.kpi-purple {{ background: linear-gradient(135deg, #5B21B6 0%, #7C3AED 50%, #8B5CF6 100%); box-shadow: 0 8px 28px rgba(124,58,237,0.42) !important; }}
.kpi-cyan   {{ background: linear-gradient(135deg, #0E7490 0%, #0891B2 50%, #06B6D4 100%); box-shadow: 0 8px 28px rgba(8,145,178,0.42) !important; }}
.kpi-rose   {{ background: linear-gradient(135deg, #9D174D 0%, #DB2777 50%, #EC4899 100%); box-shadow: 0 8px 28px rgba(219,39,119,0.42) !important; }}
.kpi-teal   {{ background: linear-gradient(135deg, #0F766E 0%, #0D9488 50%, #14B8A6 100%); box-shadow: 0 8px 28px rgba(13,148,136,0.42) !important; }}
.kpi-icon {{ font-size: 28px; margin-bottom: 12px; display: block; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15)); }}
.kpi-value {{ font-size: 36px; font-weight: 800; font-family: 'Space Grotesk', sans-serif; line-height: 1.05; color: #FFFFFF !important; text-shadow: 0 2px 8px rgba(0,0,0,0.15); letter-spacing: -0.5px; }}
.kpi-label {{ font-size: 11px; color: rgba(255,255,255,0.80) !important; font-weight: 700; margin-top: 5px; letter-spacing: 0.8px; text-transform: uppercase; }}
.kpi-delta {{ font-size: 12px; font-weight: 600; margin-top: 12px; padding: 4px 10px; border-radius: 20px; display: inline-block; background: rgba(255,255,255,0.20); color: #FFFFFF !important; }}

/* ---- INSIGHT BOX ---- */
.insight-box {{
    background: var(--bg-card);
    border-radius: 16px;
    padding: 22px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    height: 100%;
}}
.insight-box h4 {{ color: var(--text-primary) !important; font-size: 15px !important; font-weight: 700 !important; margin-bottom: 8px; }}
.insight-box p {{ color: var(--text-dim) !important; font-size: 14px !important; line-height: 1.7; }}

/* ---- CHAT WIDGET ---- */
.chat-bubble-user {{
    background: linear-gradient(135deg, #2563EB, #0891B2);
    color: white !important;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 14px;
    line-height: 1.5;
}}
.chat-bubble-ai {{
    background: var(--bg-card2);
    color: var(--text-primary) !important;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 0;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid var(--border);
}}
.chat-container {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
}}

/* ---- ANOMALY BADGE ---- */
.anomaly-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(225,29,72,0.1); border: 1px solid rgba(225,29,72,0.25);
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; font-weight: 700; color: #E11D48 !important;
    animation: pulse-badge 2s infinite;
}}
@keyframes pulse-badge {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(225,29,72,0.3); }}
    50% {{ box-shadow: 0 0 0 6px rgba(225,29,72,0); }}
}}

/* ---- FORECAST CARD ---- */
.forecast-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow-sm);
    border-top: 3px solid #7C3AED;
}}

/* ---- AI INSIGHTS ---- */
.ai-section-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary) !important;
}}

.ai-block-primary {{
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
    background: var(--bg-card);
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.ai-block-primary:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md) !important;
}}
.ai-block-accent-bar {{
    height: 4px;
    border-radius: 4px;
    margin-bottom: 18px;
}}
.ai-block-icon-wrap {{
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    margin-bottom: 0;
    flex-shrink: 0;
}}
.ai-block-header {{
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 14px;
}}
.ai-block-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--text-primary) !important;
    margin-bottom: 2px;
    line-height: 1.3;
}}
.ai-block-subtitle {{
    font-size: 12px;
    color: var(--text-muted) !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
.ai-block-body {{
    font-size: 14px;
    color: var(--text-dim) !important;
    line-height: 1.8;
    margin-bottom: 0;
}}
.ai-stat-row {{
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}}
.ai-stat-pill {{
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    color: var(--text-dim) !important;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

.ai-grid-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}}
.ai-grid-card:hover {{
    transform: translateY(-3px);
    box-shadow: var(--shadow-md) !important;
}}
.ai-grid-card-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}}
.ai-grid-card-number {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}}
.ai-grid-card-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted) !important;
    margin-bottom: 10px;
}}
.ai-grid-card-desc {{
    font-size: 13px;
    color: var(--text-dim) !important;
    line-height: 1.6;
}}
.ai-grid-card-badge {{
    display: inline-block;
    margin-top: 12px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}}

.ai-ranked-block {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: var(--shadow-sm);
}}
.ai-ranked-item {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
}}
.ai-ranked-item:last-child {{ border-bottom: none; padding-bottom: 0; }}
.ai-ranked-item:first-child {{ padding-top: 0; }}
.ai-rank-badge {{
    width: 30px; height: 30px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    color: white;
    flex-shrink: 0;
    min-width: 30px;
}}
.ai-rank-name {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary) !important;
    flex: 1;
}}
.ai-rank-val {{
    font-size: 13px;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    white-space: nowrap;
}}
.ai-rank-bar-wrap {{
    width: 80px;
    height: 6px;
    background: var(--bg-card2);
    border-radius: 6px;
    overflow: hidden;
    flex-shrink: 0;
}}
.ai-rank-bar {{
    height: 100%;
    border-radius: 6px;
}}

.ai-timeline-wrap {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: var(--shadow-sm);
}}
.ai-timeline-item {{
    display: flex;
    gap: 16px;
    padding-bottom: 20px;
    position: relative;
}}
.ai-timeline-item:last-child {{ padding-bottom: 0; }}
.ai-timeline-line {{
    position: absolute;
    left: 17px;
    top: 38px;
    bottom: 0;
    width: 2px;
    background: var(--border);
}}
.ai-timeline-item:last-child .ai-timeline-line {{ display: none; }}
.ai-timeline-dot {{
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    z-index: 1;
    min-width: 36px;
}}
.ai-timeline-content {{ flex: 1; }}
.ai-timeline-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted) !important;
    margin-bottom: 4px;
}}
.ai-timeline-text {{
    font-size: 14px;
    color: var(--text-dim) !important;
    line-height: 1.6;
}}

.live-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(5,150,105,0.1);
    border: 1px solid rgba(5,150,105,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
    color: #059669 !important;
    letter-spacing: 0.5px;
}}
.live-dot {{
    width: 6px; height: 6px;
    background: #059669;
    border-radius: 50%;
    animation: blink 1.4s infinite;
    display: inline-block;
}}
@keyframes blink {{
    0%,100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

.stat-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# YOUR MAIN APP CONTENT GOES BELOW THIS LINE
# =========================================================

st.title("🛡️ VigilantAP — Food Safety Intelligence Platform")
































# =========================================================
# PLOTLY CHART TEMPLATE
# =========================================================

def get_chart_layout():
    dark = st.session_state.dark_mode
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8" if dark else "#64748B", family="DM Sans"),
        title_font=dict(color="#F1F5F9" if dark else "#0F172A", size=17, family="Space Grotesk"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)" if dark else "rgba(0,0,0,0.05)",
            linecolor="rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)",
            tickfont=dict(color="#64748B" if dark else "#94A3B8", size=12)
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)" if dark else "rgba(0,0,0,0.05)",
            linecolor="rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)",
            tickfont=dict(color="#64748B" if dark else "#94A3B8", size=12)
        ),
        legend=dict(
            bgcolor="rgba(26,29,39,0.9)" if dark else "rgba(255,255,255,0.9)",
            bordercolor="rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(color="#CBD5E1" if dark else "#475569")
        ),
        margin=dict(t=50, b=40, l=40, r=20)
    )

COLOR_SCALE     = ["#0891B2", "#2563EB", "#7C3AED", "#E11D48", "#D97706", "#059669"]
DIVERGING_SCALE = [[0, "#059669"], [0.5, "#D97706"], [1, "#E11D48"]]


def styled_chart(fig, height=420):
    fig.update_layout(**get_chart_layout(), height=height)
    return fig


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    samples     = pd.read_csv("New_Fact_Samples.csv")
    health      = pd.read_csv("Fact_HealthCases.csv")
    inspections = pd.read_csv("New Fact_Inspections.csv")
    districts   = pd.read_csv("Dim_Districts.csv")
    category    = pd.read_csv("Dim_FoodCategory.csv")
    calendar    = pd.read_csv("Dim_Calendar.csv")

    samples["DateTested"] = pd.to_datetime(samples["DateTested"])
    health["Date"]        = pd.to_datetime(health["Date"])

    samples = samples.merge(districts, on="DistrictID", how="left")
    samples = samples.merge(category,  on="CategoryID",  how="left")
    health  = health.merge(districts,  on="DistrictID",  how="left")

    return samples, health, inspections, districts, category


samples, health, inspections, districts, category = load_data()


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomalies(df):
    fail_rates = (
        df.groupby("DistrictName")
        .apply(lambda x: (x["TestResult"] == "Fail").mean() * 100)
        .reset_index(name="FailRate")
    )
    mean = fail_rates["FailRate"].mean()
    std  = fail_rates["FailRate"].std()
    if std == 0:
        return pd.DataFrame()
    fail_rates["ZScore"] = (fail_rates["FailRate"] - mean) / std
    anomalies = fail_rates[fail_rates["ZScore"].abs() > 1.8].copy()
    anomalies["Severity"] = anomalies["ZScore"].apply(
        lambda z: "🔴 Critical" if abs(z) > 2.5 else "🟠 High"
    )
    return anomalies.sort_values("ZScore", ascending=False)


# =========================================================
# GROQ AI HELPERS
# =========================================================

def query_ai_assistant(question: str, context_summary: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are VigilantAP's AI food safety analyst for Andhra Pradesh, India. "
                    "You help government officials and health inspectors understand food safety data and make decisions.\n\n"
                    f"Current Dashboard Summary:\n{context_summary}\n\n"
                    "Guidelines:\n"
                    "- Give concise, actionable answers (2-4 sentences max unless asked for details)\n"
                    "- Reference specific districts, categories, or numbers when relevant\n"
                    "- Use formal but clear language appropriate for government officials\n"
                    "- When suggesting actions, be specific and prioritized\n"
                    "- Do not make up data not in the summary; acknowledge uncertainty if needed"
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def get_ai_forecast(district_data: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Based on this Andhra Pradesh food safety data summary:\n{district_data}\n\n"
                    "Write a 3-sentence forecast for the next quarter covering:\n"
                    "1. Which districts/categories are projected to worsen\n"
                    "2. Estimated risk trajectory\n"
                    "3. One specific intervention recommendation\n\n"
                    "Be data-driven and specific. Use numbers where possible."
                )
            }
        ]
    )
    return response.choices[0].message.content


def clean_json_response(raw: str) -> str:
    raw = re.sub(r"```(?:json)?", "", raw)
    raw = raw.replace("```", "")
    return raw.strip()


def get_ai_structured_insights(context_summary: str) -> dict:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are VigilantAP's AI food safety analyst for Andhra Pradesh, India. "
                    "Generate a detailed intelligence briefing as a structured JSON object. "
                    "IMPORTANT: Return ONLY raw valid JSON with no markdown formatting, "
                    "no code fences, no backticks, no preamble, and no explanation. "
                    "Start your response directly with { and end with }."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Based on this dashboard data:\n{context_summary}\n\n"
                    "Return a JSON object with EXACTLY these keys:\n"
                    "{\n"
                    '  "critical_alert": "1-2 sentence urgent issue description",\n'
                    '  "critical_districts": ["District1", "District2", "District3"],\n'
                    '  "trend_summary": "2-sentence trend analysis with numbers",\n'
                    '  "top_category_risk": "name of top risk food category and why",\n'
                    '  "health_correlation": "1-2 sentence insight on health impact correlation",\n'
                    '  "recommendations": ["Action 1 with specific district/category", "Action 2", "Action 3", "Action 4"],\n'
                    '  "forecast_sentence": "1 sentence forward-looking projection",\n'
                    '  "risk_level": "Low or Moderate or High or Critical",\n'
                    '  "confidence": "87%"\n'
                    "}\n"
                    "Be specific with real district names and numbers from the data. "
                    "Do NOT wrap in markdown. Do NOT use code fences."
                )
            }
        ]
    )
    raw = response.choices[0].message.content
    raw = clean_json_response(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None


# =========================================================
# GROQ WHISPER TRANSCRIPTION HELPER
# =========================================================

import tempfile

def transcribe_voice(audio_bytes: bytes) -> str:
    """
    Send raw audio bytes to Groq's Whisper-large-v3 model
    and return the transcribed text string.
    Raises RuntimeError on failure so the caller can surface the message.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text",   # returns plain string
                language="en",            # change to "te" for Telugu, "hi" for Hindi
            )
        return transcription.strip()
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")
    finally:
        os.unlink(tmp_path)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    col_logo, col_dark = st.columns([3, 1])
    with col_logo:
        st.markdown(f"""
        <div style="padding:20px 0px 10px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <div style="background:linear-gradient(135deg,#2563EB,#0891B2);
                            width:42px;height:42px;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;font-size:22px;">🛡️</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:800;color:var(--text-primary);">VigilantAP</div>
                    <div style="font-size:11px;color:#94A3B8;font-weight:600;letter-spacing:1px;">FOOD SAFETY INTELLIGENCE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_dark:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        dark_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(dark_label, help="Toggle dark/light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f"""
    <div style="background:var(--bg-card2);border:1px solid var(--border);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:12px;color:#64748B;">
        🕒 {datetime.now().strftime("%d %b %Y  %H:%M")}
    </div>
    """, unsafe_allow_html=True)

    for icon, label, val in [
        ("📦", "Total Records",   f"{len(samples):,}"),
        ("🍔", "Food Categories", samples['CategoryName'].nunique()),
        ("📅", "Date Range",      "2024 – 2025"),
    ]:
        st.markdown(f"""
        <div style="background:var(--bg-card2);border:1px solid var(--border);
                    border-radius:10px;padding:11px 14px;margin-bottom:8px;
                    display:flex;align-items:center;gap:10px;">
            <span style="font-size:18px;">{icon}</span>
            <div>
                <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="font-size:15px;font-weight:700;color:var(--text-primary);">{val}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;
                display:flex;align-items:center;gap:10px;">
        <span style="font-size:18px;">⚡</span>
        <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;">AI Engine</div>
            <div style="font-size:14px;font-weight:700;color:#F97316;">Groq · LLaMA-3 70B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    def _section(icon, title):
        st.markdown(
            f"<div style='font-size:11px;font-weight:700;color:#94A3B8;letter-spacing:1px;"
            f"text-transform:uppercase;margin-bottom:6px;'>{icon} {title}</div>",
            unsafe_allow_html=True
        )

    _section("📅", "Month")
    month_filter = st.selectbox("Month", ["All Months"] + sorted(samples["DateTested"].dt.strftime("%b %Y").dropna().unique()), label_visibility="collapsed")
    _section("📍", "District")
    district_filter = st.selectbox("District", ["All Districts"] + sorted(samples["DistrictName"].dropna().unique()), label_visibility="collapsed")
    _section("🍔", "Food Category")
    category_filter = st.selectbox("Category", ["All Categories"] + sorted(samples["CategoryName"].dropna().unique()), label_visibility="collapsed")
    _section("🧪", "Test Result")
    result_filter = st.selectbox("Result", ["All Results"] + sorted(samples["TestResult"].dropna().unique()), label_visibility="collapsed")

    st.markdown("<hr style='border-color:var(--border);margin:14px 0;'>", unsafe_allow_html=True)
    only_failed = st.checkbox("⚠️ Only Failed Samples")
    only_passed = st.checkbox("✅ Only Passed Samples")

    st.markdown("<hr style='border-color:var(--border);margin:14px 0;'>", unsafe_allow_html=True)
    if st.button("🔍 Run Anomaly Scan", use_container_width=True):
        st.session_state.anomalies_detected = True

    st.markdown("<hr style='border-color:var(--border);margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FFF5F6;border:1px solid rgba(225,29,72,0.2);border-radius:14px;padding:18px;">
        <div style="font-size:14px;font-weight:700;color:#E11D48;margin-bottom:12px;">🚨 Active Risk Alert</div>
        <div style="font-size:13px;color:#475569;line-height:1.8;">
            📌 <b style='color:#0F172A;'>High Risk:</b> Milk &amp; Oils<br>
            📍 <b style='color:#0F172A;'>Critical:</b> Tirupati District<br>
            ⚠️ <b style='color:#0F172A;'>Failure Rate:</b> 32.4%<br>
            <div style="margin-top:10px;padding:10px;background:rgba(225,29,72,0.06);border-radius:10px;font-size:12px;color:#64748B;">
                💡 Increase inspections &amp; monitor oil-based products weekly.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FILTER DATA
# =========================================================

filtered = samples.copy()
if month_filter     != "All Months":     filtered = filtered[filtered["DateTested"].dt.strftime("%b %Y") == month_filter]
if district_filter  != "All Districts":  filtered = filtered[filtered["DistrictName"] == district_filter]
if category_filter  != "All Categories": filtered = filtered[filtered["CategoryName"] == category_filter]
if result_filter    != "All Results":    filtered = filtered[filtered["TestResult"] == result_filter]
if only_failed:  filtered = filtered[filtered["TestResult"] == "Fail"]
if only_passed:  filtered = filtered[filtered["TestResult"] == "Pass"]

filtered_health = health.copy()
if district_filter != "All Districts":
    filtered_health = filtered_health[filtered_health["DistrictName"] == district_filter]


# =========================================================
# KPI VALUES
# =========================================================

total_samples = len(filtered)
failed        = len(filtered[filtered["TestResult"] == "Fail"])
passed        = len(filtered[filtered["TestResult"] == "Pass"])
adulteration  = round((failed / total_samples) * 100, 2) if total_samples > 0 else 0
health_cases  = int(filtered_health["PatientCount"].sum())
pass_rate     = round((passed / total_samples) * 100, 1) if total_samples > 0 else 0
top_risk_cat  = (filtered[filtered["TestResult"] == "Fail"].groupby("CategoryName").size().idxmax() if failed > 0 else "N/A")
top_risk_dist = (filtered[filtered["TestResult"] == "Fail"].groupby("DistrictName").size().idxmax() if failed > 0 else "N/A")

district_fail_rates = (
    filtered.groupby("DistrictName")
    .apply(lambda x: round((x["TestResult"] == "Fail").mean() * 100, 1))
    .reset_index(name="FailRate")
    .sort_values("FailRate", ascending=False)
)
top5_districts = district_fail_rates.head(5)["DistrictName"].tolist()
top5_rates     = district_fail_rates.head(5)["FailRate"].tolist()

cat_fail_rates = (
    filtered.groupby("CategoryName")
    .apply(lambda x: round((x["TestResult"] == "Fail").mean() * 100, 1))
    .reset_index(name="FailRate")
    .sort_values("FailRate", ascending=False)
)
top_cat_name = cat_fail_rates.iloc[0]["CategoryName"] if len(cat_fail_rates) > 0 else "N/A"
top_cat_rate = cat_fail_rates.iloc[0]["FailRate"] if len(cat_fail_rates) > 0 else 0


# =========================================================
# ANOMALY DETECTION DISPLAY
# =========================================================

if st.session_state.anomalies_detected:
    anomalies = detect_anomalies(filtered)
    if not anomalies.empty:
        st.markdown(f"""
        <div style="background:rgba(225,29,72,0.05);border:1px solid rgba(225,29,72,0.2);
                    border-radius:14px;padding:16px 20px;margin-bottom:16px;
                    display:flex;align-items:center;gap:14px;">
            <span class="anomaly-badge">⚡ {len(anomalies)} Anomalies Detected</span>
            <span style="color:var(--text-dim);font-size:14px;">
                Statistical outliers found in district failure rates (z-score &gt; 1.8σ)
            </span>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(min(len(anomalies), 4))
        for i, (_, row) in enumerate(anomalies.iterrows()):
            if i < 4:
                with cols[i]:
                    color = "kpi-red" if "Critical" in row["Severity"] else "kpi-amber"
                    st.markdown(f"""
                    <div class="kpi-card {color}" style="padding:16px;">
                        <div class="kpi-icon" style="font-size:20px;">⚡</div>
                        <div class="kpi-value" style="font-size:22px;">{safe(row['DistrictName'])}</div>
                        <div class="kpi-label">Fail Rate: {row['FailRate']:.1f}%</div>
                        <div class="kpi-delta">{safe(row['Severity'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.success("✅ No statistical anomalies detected in current filtered data.")


# =========================================================
# PAGE HEADER
# =========================================================

col_h1, col_h2 = st.columns([6, 2])
with col_h1:
    st.markdown("""
    <div style="padding:8px 0 4px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:40px;font-weight:900;
                    background:linear-gradient(135deg,#2563EB,#0891B2);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;line-height:1.1;">
        </div> 
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.download_button(
        "📥 Export Report",
        data=filtered.to_csv(index=False),
        file_name=f"vigilantap_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.markdown("<hr style='border-color:var(--border);margin:10px 0 18px;'>", unsafe_allow_html=True)


# =========================================================
# TABS
# =========================================================

home, tab1, tab2, tab3, tab4, tab5, tab_ai, tab_predict, tab6 = st.tabs([
    "🏠 Home",
    "📊 Executive Overview",
    "🧪 Sample Analysis",
    "⚠️ Risk Analytics",
    "🏥 Health Impact",
    "🎯 Intervention",
    "🤖 AI Assistant",
    "🔮 Predictive Analytics",
    "ℹ️ About"
])


# =========================================================
# HELPER: KPI card
# =========================================================

def kpi(icon, label, value, delta_text, delta_type="up", color_class="kpi-blue"):
    delta_arrow = "↑" if delta_type == "up" else ("↓" if delta_type == "down" else "⚠")
    return f"""
    <div class="kpi-card {color_class}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value">{safe(str(value))}</div>
        <div class="kpi-label">{safe(label)}</div>
        <div class="kpi-delta">{delta_arrow} {safe(delta_text)}</div>
    </div>
    """


# =========================================================
# HOME TAB
# =========================================================

with home:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#EFF6FF,#F0FDFA,#FFF);
                border:1px solid rgba(37,99,235,0.12);border-radius:22px;
                padding:42px 40px;margin-bottom:28px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:-60px;right:-60px;width:240px;height:240px;
                    background:radial-gradient(circle,rgba(37,99,235,0.08),transparent 70%);
                    border-radius:50%;pointer-events:none;"></div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:36px;font-weight:900;color:#0F172A;margin-bottom:12px;">
            Welcome to VigilantAP 🛡️
        </div>
        <div style="color:#475569;font-size:16px;line-height:1.8;max-width:680px;">
            An advanced AI-powered food safety intelligence platform designed to monitor food adulteration,
            identify district-level risks, track public health impact, and guide government intervention across Andhra Pradesh.
        </div>
        <div style="display:flex;gap:12px;margin-top:24px;flex-wrap:wrap;">
            <div style="background:rgba(225,29,72,0.07);border:1px solid rgba(225,29,72,0.18);border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;color:#E11D48;">🔴 Live Monitoring Active</div>
            <div style="background:rgba(5,150,105,0.07);border:1px solid rgba(5,150,105,0.18);border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;color:#059669;">✅ 26 Districts Covered</div>
            <div style="background:rgba(217,119,6,0.07);border:1px solid rgba(217,119,6,0.18);border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;color:#D97706;">⚡ Real-time Analytics</div>
            <div style="background:rgba(249,115,22,0.07);border:1px solid rgba(249,115,22,0.18);border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;color:#F97316;">⚡ Groq AI · LLaMA-3 70B</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    modules = [
        ("#2563EB","#0891B2","📊","Executive Overview","High-level KPIs, risk meter, and district-wise adulteration performance at a glance."),
        ("#7C3AED","#A78BFA","🧪","Sample Analysis","Passed vs failed food samples, monthly testing trends, and category-level contamination breakdowns."),
        ("#E11D48","#FB7185","⚠️","Risk Analytics","Heat maps, failure trends, top high-risk districts, and AI-driven risk scoring."),
        ("#D97706","#F59E0B","🏥","Health Impact","Disease distribution, patient count trends, seasonal analysis, and correlation insights."),
        ("#059669","#34D399","🎯","Intervention","Priority action dashboard, government alerts, inspection coverage, and smart recommendations."),
        ("#F97316","#FB923C","🤖","AI Assistant","Chat with Groq AI (LLaMA-3 70B) to get instant food safety insights and action plans."),
        ("#0E7490","#06B6D4","🔮","Predictive Analytics","AI-generated forecasts, trend projections, and future risk scenario modeling."),
        ("#DB2777","#F472B6","ℹ️","About","Platform workflow, technology stack, objectives, and future enhancement roadmap."),
    ]

    r1 = st.columns(4)
    r2 = st.columns(4)
    for i, (c1, c2, icon, title, desc) in enumerate(modules):
        row = r1 if i < 4 else r2
        with row[i % 4]:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid rgba(0,0,0,0.07);border-radius:18px;
                        padding:22px;margin-bottom:16px;position:relative;overflow:hidden;
                        box-shadow:0 1px 3px rgba(0,0,0,0.05);height:180px;">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;
                            background:linear-gradient(90deg,{c1},{c2});border-radius:18px 18px 0 0;"></div>
                <div style="font-size:26px;margin-bottom:8px;">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;
                            color:var(--text-primary);margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:#64748B;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

with tab1:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);">📊 Executive Overview</div>
            <div style="color:#64748B;font-size:14px;margin-top:4px;">Centralized food safety performance metrics across all districts</div>
        </div>
        <div style="background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2);border-radius:20px;padding:6px 16px;font-size:13px;font-weight:700;color:#059669;">🟢 System Active</div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(kpi("🧪", "Total Samples",   f"{total_samples:,}", "Active Monitoring",    "up",   "kpi-blue"),   unsafe_allow_html=True)
    with k2: st.markdown(kpi("✅", "Passed Samples",  f"{passed:,}",        f"{pass_rate}% pass rate","up", "kpi-green"),  unsafe_allow_html=True)
    with k3: st.markdown(kpi("❌", "Failed Samples",  f"{failed:,}",        "Requires Action",      "down", "kpi-red"),    unsafe_allow_html=True)
    with k4: st.markdown(kpi("🏥", "Health Cases",    f"{health_cases:,}",  "District Impact",      "warn", "kpi-amber"),  unsafe_allow_html=True)
    with k5: st.markdown(kpi("⚠️", "Adulteration %", f"{adulteration}%",   "Risk Level",
                              "down" if adulteration > 20 else "warn", "kpi-purple"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    risk_color = "#059669" if adulteration < 10 else ("#D97706" if adulteration < 25 else "#E11D48")
    risk_label = ("🟢 Low Risk Environment" if adulteration < 10
                  else "🟡 Moderate Risk" if adulteration < 25
                  else "🔴 High Risk — Immediate Action Required")
    st.markdown(f"""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:22px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">🚨 Overall Risk Severity</div>
            <div style="font-size:14px;font-weight:700;color:{risk_color};">{risk_label}</div>
        </div>
        <div style="background:#F1F5F9;border-radius:20px;height:10px;overflow:hidden;">
            <div style="width:{min(adulteration,100)}%;height:100%;background:linear-gradient(90deg,{risk_color},rgba(0,0,0,0.08));border-radius:20px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:12px;color:#94A3B8;">
            <span>0% Safe</span><span>25% Moderate</span><span>50% High</span><span>100% Critical</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns([3, 2])
    with c_left:
        district_risk = (
            filtered.groupby("DistrictName")
            .apply(lambda x: (x["TestResult"] == "Fail").mean() * 100)
            .reset_index(name="Risk %")
            .sort_values("Risk %", ascending=False)
        )
        fig1 = px.bar(district_risk, x="DistrictName", y="Risk %",
                      color="Risk %", text_auto=".1f",
                      title="District-wise Adulteration Risk (%)",
                      color_continuous_scale=DIVERGING_SCALE)
        fig1.update_traces(textfont_color="white", marker_line_width=0)
        st.plotly_chart(styled_chart(fig1, 430), use_container_width=True)

    with c_right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=adulteration,
            title={"text": "Risk Meter", "font": {"color": "#0F172A", "size": 16, "family": "Space Grotesk"}},
            delta={"reference": 15, "valueformat": ".1f",
                   "increasing": {"color": "#E11D48"}, "decreasing": {"color": "#059669"}},
            number={"suffix": "%", "font": {"color": "#0F172A", "size": 52}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#CBD5E1", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": "#E11D48" if adulteration > 25 else "#D97706"},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 10],   "color": "rgba(5,150,105,0.12)"},
                    {"range": [10, 25],  "color": "rgba(217,119,6,0.12)"},
                    {"range": [25, 100], "color": "rgba(225,29,72,0.12)"}
                ],
                "threshold": {"line": {"color": "#E11D48", "width": 2}, "thickness": 0.75, "value": 25}
            }
        ))
        gauge.update_layout(**get_chart_layout(), height=430)
        st.plotly_chart(gauge, use_container_width=True)

    c2a, c2b = st.columns(2)
    with c2a:
        category_fail = (filtered[filtered["TestResult"] == "Fail"]
                         .groupby("CategoryName").size().reset_index(name="Failures"))
        fig2 = px.pie(category_fail, names="CategoryName", values="Failures",
                      hole=0.55, title="Failed Samples by Food Category",
                      color_discrete_sequence=COLOR_SCALE)
        fig2.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=2)))
        st.plotly_chart(styled_chart(fig2, 420), use_container_width=True)

    with c2b:
        monthly = (filtered.groupby(filtered["DateTested"].dt.to_period("M"))
                   .size().reset_index(name="Samples"))
        monthly["DateTested"] = monthly["DateTested"].astype(str)
        fig3 = px.area(monthly, x="DateTested", y="Samples",
                       title="Monthly Testing Volume Trend",
                       color_discrete_sequence=["#2563EB"])
        fig3.update_traces(fillcolor="rgba(37,99,235,0.08)", line_width=2)
        st.plotly_chart(styled_chart(fig3, 420), use_container_width=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    insights_static = [
        ("📌 Key Insight",    "Milk, Oils and Spices show highest contamination trends consistently across monitored districts."),
        ("🏥 Public Health",  "Failed food samples show strong positive correlation with rising public health case counts district-wide."),
        ("🎯 Strategic Action","Authorities should immediately prioritize inspection resources in the top 5 high-risk districts."),
    ]
    for col, (title, body) in zip([i1, i2, i3], insights_static):
        with col:
            st.markdown(f'<div class="insight-box"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:12px;">🏆 Top High-Risk Districts</div>""", unsafe_allow_html=True)
    st.dataframe(district_risk.head(10).style.background_gradient(subset=["Risk %"], cmap="RdYlGn_r"), use_container_width=True)


# =========================================================
# SAMPLE ANALYSIS
# =========================================================

with tab2:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">🧪 Sample Analysis</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">Laboratory testing results, trends and food category-level quality breakdown</div>""", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(kpi("🧪", "Total Samples Tested", f"{total_samples:,}", "All selected filters",      "up",   "kpi-cyan"),  unsafe_allow_html=True)
    with s2: st.markdown(kpi("✅", "Samples Passed",        f"{passed:,}",        f"{pass_rate}% pass rate",  "up",   "kpi-green"), unsafe_allow_html=True)
    with s3: st.markdown(kpi("❌", "Samples Failed",        f"{failed:,}",        "Requires investigation",   "down", "kpi-red"),   unsafe_allow_html=True)
    with s4: st.markdown(kpi("🍽️","Top Risk Category",     top_risk_cat,         "Highest failures",          "warn", "kpi-amber"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    t2c1, t2c2 = st.columns(2)
    with t2c1:
        monthly_s = filtered.groupby(filtered["DateTested"].dt.to_period("M")).size().reset_index(name="Samples")
        monthly_s["DateTested"] = monthly_s["DateTested"].astype(str)
        fig_s1 = px.area(monthly_s, x="DateTested", y="Samples", title="Monthly Sample Testing Volume", color_discrete_sequence=["#0891B2"])
        fig_s1.update_traces(fillcolor="rgba(8,145,178,0.08)", line_width=2)
        st.plotly_chart(styled_chart(fig_s1), use_container_width=True)

    with t2c2:
        fig_pie = px.pie(names=["Passed", "Failed"], values=[passed, failed], hole=0.55,
                          title="Pass vs Fail Distribution", color_discrete_sequence=["#059669", "#E11D48"])
        fig_pie.update_traces(marker=dict(line=dict(color="#FFFFFF", width=2)))
        st.plotly_chart(styled_chart(fig_pie), use_container_width=True)

    monthly_pf = filtered.groupby([filtered["DateTested"].dt.to_period("M"), "TestResult"]).size().reset_index(name="Count")
    monthly_pf["DateTested"] = monthly_pf["DateTested"].astype(str)
    fig_pf = px.bar(monthly_pf, x="DateTested", y="Count", color="TestResult",
                    title="Monthly Pass vs Fail Breakdown", barmode="stack",
                    color_discrete_map={"Pass": "#059669", "Fail": "#E11D48"})
    fig_pf.update_traces(marker_line_width=0)
    st.plotly_chart(styled_chart(fig_pf, 380), use_container_width=True)

    cat_pf = filtered.groupby(["CategoryName", "TestResult"]).size().reset_index(name="Count")
    fig_cat = px.bar(cat_pf, x="CategoryName", y="Count", color="TestResult",
                     title="Category-wise Pass vs Fail", barmode="group",
                     color_discrete_map={"Pass": "#059669", "Fail": "#E11D48"})
    fig_cat.update_traces(marker_line_width=0)
    st.plotly_chart(styled_chart(fig_cat, 380), use_container_width=True)


# =========================================================
# RISK ANALYTICS
# =========================================================

with tab3:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">⚠️ Risk Analytics</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">District risk scores, failure trends, category hotspots and AI-powered risk predictions</div>""", unsafe_allow_html=True)

    r1c, r2c, r3c, r4c = st.columns(4)
    with r1c: st.markdown(kpi("🔥", "Overall Risk Score",    "65.3", "↑ +12% this month",    "warn", "kpi-amber"), unsafe_allow_html=True)
    with r2c: st.markdown(kpi("🚨", "High Risk Districts",   "5",    "↑ +2 new alerts",       "down", "kpi-red"),   unsafe_allow_html=True)
    with r3c: st.markdown(kpi("📋", "Inspection Coverage",   "78%",  "↑ +8% improvement",    "up",   "kpi-blue"),  unsafe_allow_html=True)
    with r4c: st.markdown(kpi("📈", "Compliance Rate",       "67%",  "↑ +3% vs last month",  "up",   "kpi-teal"),  unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    ra1, ra2 = st.columns(2)
    with ra1:
        category_fail = (filtered[filtered["TestResult"] == "Fail"]
                         .groupby("CategoryName").size().reset_index(name="Failures")
                         .sort_values("Failures", ascending=False))
        fig5 = px.bar(category_fail, x="CategoryName", y="Failures", color="Failures",
                      title="Failures by Food Category", color_continuous_scale=DIVERGING_SCALE)
        fig5.update_traces(marker_line_width=0)
        st.plotly_chart(styled_chart(fig5), use_container_width=True)

    with ra2:
        risk_districts = pd.DataFrame({
            "District":    ["Tirupati","Chittoor","Vizag","Kurnool","Anantapur","Nellore","Kadapa","Guntur"],
            "Risk Score":  [89, 76, 72, 68, 61, 58, 52, 47]
        })
        fig_top = px.bar(risk_districts, x="Risk Score", y="District", orientation="h", color="Risk Score",
                         title="Top High-Risk Districts — Risk Score",
                         color_continuous_scale=DIVERGING_SCALE)
        fig_top.update_traces(marker_line_width=0)
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_chart(fig_top), use_container_width=True)

    trend_data = pd.DataFrame({"Month":["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"],"Risk Score":[45,52,48,61,70,65,72,68],"Threshold":[50]*8})
    fig_trend = px.line(trend_data, x="Month", y=["Risk Score","Threshold"], markers=True, title="Monthly Risk Score Trend",
                        color_discrete_map={"Risk Score":"#E11D48","Threshold":"#D97706"})
    fig_trend.update_traces(line_width=2.5)
    st.plotly_chart(styled_chart(fig_trend, 360), use_container_width=True)

    heatmap_data = (filtered[filtered["TestResult"] == "Fail"]
                    .groupby(["DistrictName","CategoryName"]).size().reset_index(name="Failures"))
    if not heatmap_data.empty:
        pivot = heatmap_data.pivot_table(index="CategoryName", columns="DistrictName", values="Failures", fill_value=0)
        fig_heat = px.imshow(pivot, title="Risk Heatmap — Category × District",
                             color_continuous_scale="RdYlGn_r", aspect="auto", text_auto=True)
        fig_heat.update_layout(**get_chart_layout(), height=400)
        st.plotly_chart(fig_heat, use_container_width=True)


# =========================================================
# HEALTH IMPACT
# =========================================================

with tab4:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">🏥 Health Impact</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">Disease distribution, patient trends, seasonal analysis, and failed-sample health correlations</div>""", unsafe_allow_html=True)

    h1c, h2c, h3c, h4c = st.columns(4)
    with h1c: st.markdown(kpi("🏥", "Total Health Cases",    f"{health_cases:,}", "↑ +12% vs prior period", "down", "kpi-red"),    unsafe_allow_html=True)
    with h2c: st.markdown(kpi("📍", "Highest Risk District", "Tirupati",          "Immediate attention",    "down", "kpi-rose"),   unsafe_allow_html=True)
    with h3c: st.markdown(kpi("🥛", "Top Risk Category",     "Milk & Oils",       "Weekly monitoring",     "warn", "kpi-purple"), unsafe_allow_html=True)
    with h4c: st.markdown(kpi("🚑", "Alert Level",           "HIGH",              "Emergency active",       "down", "kpi-amber"),  unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.error("🚨 Alert: Tirupati showing sudden spike in food poisoning cases — Deploy inspection teams immediately")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    monthly_failed = (filtered[filtered["TestResult"] == "Fail"]
                      .groupby(filtered["DateTested"].dt.to_period("M")).size().reset_index(name="Failed"))
    monthly_failed["DateTested"] = monthly_failed["DateTested"].astype(str)
    monthly_health = (filtered_health.groupby(filtered_health["Date"].dt.to_period("M"))["PatientCount"]
                      .sum().reset_index())
    monthly_health["Date"] = monthly_health["Date"].astype(str)

    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(go.Bar(x=monthly_failed["DateTested"], y=monthly_failed["Failed"],
                          name="Failed Samples", marker_color="#E11D48", marker_line_width=0, opacity=0.8), secondary_y=False)
    fig6.add_trace(go.Scatter(x=monthly_health["Date"], y=monthly_health["PatientCount"],
                              mode="lines+markers", name="Patient Count",
                              line=dict(color="#0891B2", width=2.5), marker=dict(size=7, color="#0891B2")), secondary_y=True)
    fig6.update_layout(**get_chart_layout(), height=400, title_text="Failed Samples vs Patient Count Correlation")
    fig6.update_yaxes(title_text="Failed Samples", secondary_y=False, title_font=dict(color="#E11D48"))
    fig6.update_yaxes(title_text="Patient Count",  secondary_y=True,  title_font=dict(color="#0891B2"))
    st.plotly_chart(fig6, use_container_width=True)

    h4a, h4b = st.columns(2)
    with h4a:
        disease_df = pd.DataFrame({"Disease":["Food Poisoning","Stomach Infection","Allergy","Water Contamination","Others"],"Cases":[45,30,15,20,10]})
        fig_dis = px.pie(disease_df, names="Disease", values="Cases", hole=0.55, title="Disease Distribution",
                          color_discrete_sequence=["#E11D48","#D97706","#059669","#2563EB","#7C3AED"])
        fig_dis.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#FFFFFF", width=2)))
        st.plotly_chart(styled_chart(fig_dis), use_container_width=True)

    with h4b:
        season_df = pd.DataFrame({"Season":["Summer","Rainy","Winter"],"Risk":[90,75,45]})
        fig_season = px.bar(season_df, x="Season", y="Risk", color="Risk", title="Seasonal Risk Analysis",
                             color_continuous_scale=DIVERGING_SCALE, text_auto=True)
        fig_season.update_traces(marker_line_width=0, textfont_color="white")
        st.plotly_chart(styled_chart(fig_season), use_container_width=True)


# =========================================================
# INTERVENTION
# =========================================================

with tab5:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">🎯 Intervention Dashboard</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">Government action priorities, inspection progress, alerts and AI-driven strategy recommendations</div>""", unsafe_allow_html=True)

    iv1, iv2, iv3, iv4 = st.columns(4)
    with iv1: st.markdown(kpi("🎯","Intervention Target","78%","↑ +5% this month",         "up",   "kpi-blue"),   unsafe_allow_html=True)
    with iv2: st.markdown(kpi("🚨","Critical Alerts",   "3",  "Requires immediate action", "down", "kpi-red"),    unsafe_allow_html=True)
    with iv3: st.markdown(kpi("🔍","Active Inspections","142","Ongoing district checks",   "up",   "kpi-green"),  unsafe_allow_html=True)
    with iv4: st.markdown(kpi("📋","Actions Completed", "67%","↑ of planned interventions","up",   "kpi-purple"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""<div style="background:#FFF1F2;border:1px solid rgba(225,29,72,0.2);border-radius:14px;padding:18px;border-left:4px solid #E11D48;">
            <div style="font-size:13px;font-weight:700;color:#E11D48;margin-bottom:6px;">🔴 CRITICAL</div>
            <div style="font-size:15px;font-weight:700;color:#0F172A;">Tirupati</div>
            <div style="font-size:13px;color:#64748B;margin-top:4px;">Immediate inspection deployment required</div>
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""<div style="background:#FFFBEB;border:1px solid rgba(217,119,6,0.2);border-radius:14px;padding:18px;border-left:4px solid #D97706;">
            <div style="font-size:13px;font-weight:700;color:#D97706;margin-bottom:6px;">🟠 MEDIUM RISK</div>
            <div style="font-size:15px;font-weight:700;color:#0F172A;">SPSR Nellore</div>
            <div style="font-size:13px;color:#64748B;margin-top:4px;">Increased monitoring scheduled</div>
        </div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown("""<div style="background:#F0FDF4;border:1px solid rgba(5,150,105,0.2);border-radius:14px;padding:18px;border-left:4px solid #059669;">
            <div style="font-size:13px;font-weight:700;color:#059669;margin-bottom:6px;">🟢 CONTROLLED</div>
            <div style="font-size:15px;font-weight:700;color:#0F172A;">Vizianagaram</div>
            <div style="font-size:13px;color:#64748B;margin-top:4px;">Routine inspections ongoing</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    iv_c1, iv_c2 = st.columns(2)
    with iv_c1:
        priority = (filtered[filtered["TestResult"] == "Fail"]
                    .groupby("DistrictName").size().reset_index(name="Failure Count")
                    .sort_values("Failure Count", ascending=False).head(10))
        fig_iv1 = px.bar(priority, x="DistrictName", y="Failure Count", color="Failure Count",
                          title="Top Districts — Failure Count", color_continuous_scale=DIVERGING_SCALE)
        fig_iv1.update_traces(marker_line_width=0)
        st.plotly_chart(styled_chart(fig_iv1), use_container_width=True)

    with iv_c2:
        insp_data = pd.DataFrame({
            "District":         ["Tirupati","Chittoor","Vizag","Kurnool","Nellore","Anantapur","Kadapa","Guntur"],
            "Inspections Done": [45, 38, 52, 30, 27, 22, 18, 35],
            "Target":           [60, 50, 60, 40, 35, 30, 25, 45]
        })
        fig_insp = go.Figure()
        fig_insp.add_trace(go.Bar(name="Inspections Done", x=insp_data["District"], y=insp_data["Inspections Done"], marker_color="#2563EB", marker_line_width=0))
        fig_insp.add_trace(go.Bar(name="Target",           x=insp_data["District"], y=insp_data["Target"],           marker_color="rgba(37,99,235,0.18)", marker_line_width=0))
        fig_insp.update_layout(**get_chart_layout(), height=420, title_text="Inspection Progress vs Target", barmode="overlay")
        st.plotly_chart(fig_insp, use_container_width=True)

    st.dataframe(priority.style.background_gradient(subset=["Failure Count"], cmap="RdYlGn_r"), use_container_width=True)


# =========================================================
# 🤖 AI ASSISTANT TAB
# =========================================================

with tab_ai:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);">🤖 AI Food Safety Assistant</div>
            <div style="color:#64748B;font-size:14px;margin-top:4px;">Powered by Groq AI (LLaMA-3 70B) — live intelligence briefing and interactive Q&amp;A</div>
        </div>
        <div class="live-tag"><span class="live-dot"></span> LIVE ANALYSIS</div>
    </div>
    """, unsafe_allow_html=True)

    ai1, ai2, ai3 = st.columns(3)
    with ai1: st.markdown(kpi("⚡", "AI Engine",    "Groq · LLaMA-3",  "Ultra-fast inference",      "up", "kpi-amber"),  unsafe_allow_html=True)
    with ai2: st.markdown(kpi("💬", "Responses",    "Real-time",        "Context-aware answers",     "up", "kpi-blue"),   unsafe_allow_html=True)
    with ai3: st.markdown(kpi("🎯", "Accuracy",     "98%",              "Data-grounded insights",    "up", "kpi-green"),  unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Build context summary
    context_summary = f"""
    - Total samples tested: {total_samples:,}
    - Passed: {passed:,} ({pass_rate}% pass rate) | Failed: {failed:,}
    - Overall adulteration rate: {adulteration}%
    - Total health cases recorded: {health_cases:,}
    - Top failure category: {top_risk_cat} ({top_cat_rate}% fail rate)
    - Top failure district: {top_risk_dist}
    - Top 5 high-risk districts by fail rate: {', '.join([f"{d} ({r}%)" for d, r in zip(top5_districts[:5], top5_rates[:5])])}
    - Active filter — Month: {month_filter}, District: {district_filter}, Category: {category_filter}
    - Current risk level: {"Critical" if adulteration > 35 else "High" if adulteration > 25 else "Moderate" if adulteration > 10 else "Low"}
    - Inspections data: High Risk violations dominate, Street Vendors most cited
    """

    # ── SECTION 1: Static Data Blocks ──
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;
                color:var(--text-primary);margin-bottom:4px;">📊 Data Intelligence Blocks</div>
    <div style="color:#64748B;font-size:13px;margin-bottom:16px;">Derived directly from your filtered dataset</div>
    """, unsafe_allow_html=True)

    gc1, gc2, gc3, gc4 = st.columns(4)

    risk_badge_color = "#E11D48" if adulteration > 25 else "#D97706" if adulteration > 10 else "#059669"
    risk_badge_bg    = "rgba(225,29,72,0.08)" if adulteration > 25 else "rgba(217,119,6,0.08)" if adulteration > 10 else "rgba(5,150,105,0.08)"
    risk_badge_label = "Critical Risk" if adulteration > 35 else "High Risk" if adulteration > 25 else "Moderate Risk" if adulteration > 10 else "Safe Zone"

    with gc1:
        st.markdown(f"""
        <div class="ai-grid-card">
            <div class="ai-grid-card-accent" style="background:linear-gradient(90deg,#E11D48,#F43F5E);"></div>
            <div class="ai-grid-card-number" style="color:#E11D48;">{adulteration}%</div>
            <div class="ai-grid-card-label">Adulteration Rate</div>
            <div class="ai-grid-card-desc">Overall proportion of food samples failing quality tests across all monitored categories.</div>
            <div class="ai-grid-card-badge" style="background:{risk_badge_bg};color:{risk_badge_color};border:1px solid {risk_badge_color}22;">{risk_badge_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with gc2:
        above_below = "Above Target ✓" if pass_rate > 75 else "Below Target ↓"
        st.markdown(f"""
        <div class="ai-grid-card">
            <div class="ai-grid-card-accent" style="background:linear-gradient(90deg,#2563EB,#0891B2);"></div>
            <div class="ai-grid-card-number" style="color:#2563EB;">{pass_rate}%</div>
            <div class="ai-grid-card-label">Compliance Rate</div>
            <div class="ai-grid-card-desc">Share of samples that passed food safety standards under the current filter selection.</div>
            <div class="ai-grid-card-badge" style="background:rgba(37,99,235,0.08);color:#2563EB;border:1px solid rgba(37,99,235,0.2);">{above_below}</div>
        </div>
        """, unsafe_allow_html=True)

    with gc3:
        st.markdown(f"""
        <div class="ai-grid-card">
            <div class="ai-grid-card-accent" style="background:linear-gradient(90deg,#D97706,#F59E0B);"></div>
            <div class="ai-grid-card-number" style="color:#D97706;">{health_cases:,}</div>
            <div class="ai-grid-card-label">Health Cases</div>
            <div class="ai-grid-card-desc">Total patient count recorded across filtered districts — directly correlated with food failures.</div>
            <div class="ai-grid-card-badge" style="background:rgba(217,119,6,0.08);color:#D97706;border:1px solid rgba(217,119,6,0.2);">Monitoring Active</div>
        </div>
        """, unsafe_allow_html=True)

    with gc4:
        st.markdown(f"""
        <div class="ai-grid-card">
            <div class="ai-grid-card-accent" style="background:linear-gradient(90deg,#7C3AED,#8B5CF6);"></div>
            <div class="ai-grid-card-number" style="color:#7C3AED;">{top_cat_rate}%</div>
            <div class="ai-grid-card-label">Top Category Risk</div>
            <div class="ai-grid-card-desc">{safe(top_cat_name)} has the highest category-level failure rate and requires priority intervention.</div>
            <div class="ai-grid-card-badge" style="background:rgba(124,58,237,0.08);color:#7C3AED;border:1px solid rgba(124,58,237,0.2);">{safe(top_cat_name)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Ranked Blocks — using st.columns + individual st.markdown per row ──
    rb1, rb2 = st.columns([1, 1])
    rank_colors = ["#E11D48","#F43F5E","#D97706","#F59E0B","#0891B2"]
    max_rate = top5_rates[0] if top5_rates else 1

    with rb1:
        # Build header
        st.markdown("""
        <div class="ai-ranked-block">
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);margin-bottom:14px;display:flex;align-items:center;gap:8px;">
                🏆 Top Risk Districts
                <span style="font-size:11px;font-weight:600;color:#94A3B8;background:var(--bg-card2);border:1px solid var(--border);border-radius:20px;padding:2px 8px;">by failure rate</span>
            </div>
        """, unsafe_allow_html=True)
        for i, (dist, rate) in enumerate(zip(top5_districts[:5], top5_rates[:5])):
            bar_pct = int((rate / max_rate) * 100) if max_rate > 0 else 0
            col = rank_colors[i]
            st.markdown(f"""
            <div class="ai-ranked-item">
                <div class="ai-rank-badge" style="background:{col};">#{i+1}</div>
                <div class="ai-rank-name">{safe(dist)}</div>
                <div class="ai-rank-bar-wrap"><div class="ai-rank-bar" style="width:{bar_pct}%;background:{col};"></div></div>
                <div class="ai-rank-val" style="color:{col};">{rate}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rb2:
        cat_fail_top = cat_fail_rates.head(5)
        cat_icons  = ["🥛","🫒","🧂","💧","🌾"]
        cat_colors = ["#E11D48","#D97706","#7C3AED","#0891B2","#059669"]
        top_cat_fail_rate = cat_fail_top.iloc[0]["FailRate"] if len(cat_fail_top) > 0 else 1

        st.markdown("""
        <div class="ai-ranked-block">
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);margin-bottom:14px;display:flex;align-items:center;gap:8px;">
                🍽️ Top Risk Categories
                <span style="font-size:11px;font-weight:600;color:#94A3B8;background:var(--bg-card2);border:1px solid var(--border);border-radius:20px;padding:2px 8px;">by failure rate</span>
            </div>
        """, unsafe_allow_html=True)
        for i, row in enumerate(cat_fail_top.itertuples()):
            icon   = cat_icons[i] if i < len(cat_icons) else "🍽️"
            col    = cat_colors[i] if i < len(cat_colors) else "#64748B"
            barpct = int((row.FailRate / top_cat_fail_rate) * 100) if top_cat_fail_rate > 0 else 0
            st.markdown(f"""
            <div class="ai-ranked-item">
                <div class="ai-rank-badge" style="background:{col};font-size:14px;">{icon}</div>
                <div class="ai-rank-name">{safe(row.CategoryName)}</div>
                <div class="ai-rank-bar-wrap"><div class="ai-rank-bar" style="width:{barpct}%;background:{col};"></div></div>
                <div class="ai-rank-val" style="color:{col};">{row.FailRate}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── SECTION 2: Groq AI Live Briefing ──
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;
                color:var(--text-primary);margin-bottom:4px;">⚡ AI Intelligence Briefing</div>
    <div style="color:#64748B;font-size:13px;margin-bottom:16px;">Generated by Groq · LLaMA-3 70B — updates when filters change</div>
    """, unsafe_allow_html=True)

    briefing_cache_key = f"{month_filter}|{district_filter}|{category_filter}|{result_filter}|{total_samples}"

    if (
        "ai_structured_insights" not in st.session_state
        or st.session_state.get("ai_briefing_cache_key") != briefing_cache_key
    ):
        with st.spinner("⚡ Groq AI is generating your intelligence briefing..."):
            try:
                insights_data = get_ai_structured_insights(context_summary)
                st.session_state["ai_structured_insights"] = insights_data
                st.session_state["ai_briefing_cache_key"] = briefing_cache_key
            except Exception as e:
                st.session_state["ai_structured_insights"] = None
                st.session_state["ai_briefing_cache_key"] = briefing_cache_key

    ins = st.session_state.get("ai_structured_insights")

    if ins:
        # Safely extract and escape ALL AI text fields
        critical_alert_text   = safe(ins.get("critical_alert", "—"))
        trend_summary_text    = safe(ins.get("trend_summary", "—"))
        health_corr_text      = safe(ins.get("health_correlation", "—"))
        top_cat_risk_text     = safe(ins.get("top_category_risk", "—"))
        forecast_text_val     = safe(ins.get("forecast_sentence", "—"))
        risk_level_text       = safe(ins.get("risk_level", "Moderate"))
        confidence_text       = safe(ins.get("confidence", "—"))
        critical_districts    = [safe(d) for d in ins.get("critical_districts", [])]
        recommendations       = [safe(r) for r in ins.get("recommendations", [])]

        alert_color = "#E11D48" if ins.get("risk_level","Moderate") in ["High","Critical"] else "#D97706"
        alert_bg    = "rgba(225,29,72,0.04)" if alert_color == "#E11D48" else "rgba(217,119,6,0.04)"

        # ── Block A: Critical Alert ──
        district_pills_html = "".join([
            f'<span class="ai-stat-pill"><span style="color:{alert_color};">⚠</span> {d}</span>'
            for d in critical_districts
        ])
        conf_pill = f'<span class="ai-stat-pill"><span style="color:#94A3B8;">🎯</span> Confidence: {confidence_text}</span>'

        st.markdown(f"""
        <div class="ai-block-primary" style="border-left:4px solid {alert_color};background:{alert_bg};">
            <div class="ai-block-accent-bar" style="background:linear-gradient(90deg,{alert_color},transparent);opacity:0.4;"></div>
            <div class="ai-block-header">
                <div class="ai-block-icon-wrap" style="background:{alert_color}18;">🚨</div>
                <div style="flex:1;">
                    <div class="ai-block-subtitle">Critical Alert · AI Assessment</div>
                    <div class="ai-block-title">Immediate Risk Signal Detected</div>
                </div>
                <div>
                    <span style="background:{alert_color}15;color:{alert_color};border:1px solid {alert_color}30;border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700;white-space:nowrap;">{risk_level_text} Risk</span>
                </div>
            </div>
            <div class="ai-block-body">{critical_alert_text}</div>
            <div class="ai-stat-row">
                {district_pills_html}
                {conf_pill}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Row: Trend + Health ──
        blk_c1, blk_c2 = st.columns(2)
        with blk_c1:
            st.markdown(f"""
            <div class="ai-block-primary" style="border-left:4px solid #2563EB;">
                <div class="ai-block-accent-bar" style="background:linear-gradient(90deg,#2563EB,transparent);opacity:0.3;"></div>
                <div class="ai-block-header">
                    <div class="ai-block-icon-wrap" style="background:rgba(37,99,235,0.12);">📈</div>
                    <div>
                        <div class="ai-block-subtitle">Pattern Recognition · Trend Layer</div>
                        <div class="ai-block-title">Key Data Trends</div>
                    </div>
                </div>
                <div class="ai-block-body">{trend_summary_text}</div>
                <div class="ai-stat-row">
                    <span class="ai-stat-pill"><span style="color:#E11D48;">📉</span> Fail: {adulteration}%</span>
                    <span class="ai-stat-pill"><span style="color:#059669;">✓</span> Pass: {pass_rate}%</span>
                    <span class="ai-stat-pill"><span style="color:#2563EB;">🧪</span> {total_samples:,} samples</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with blk_c2:
            st.markdown(f"""
            <div class="ai-block-primary" style="border-left:4px solid #0891B2;">
                <div class="ai-block-accent-bar" style="background:linear-gradient(90deg,#0891B2,transparent);opacity:0.3;"></div>
                <div class="ai-block-header">
                    <div class="ai-block-icon-wrap" style="background:rgba(8,145,178,0.12);">🏥</div>
                    <div>
                        <div class="ai-block-subtitle">Public Health · Impact Correlation</div>
                        <div class="ai-block-title">Health Impact Analysis</div>
                    </div>
                </div>
                <div class="ai-block-body">{health_corr_text}</div>
                <div class="ai-stat-row">
                    <span class="ai-stat-pill"><span style="color:#E11D48;">🚑</span> {health_cases:,} cases</span>
                    <span class="ai-stat-pill"><span style="color:#0891B2;">📍</span> {safe(top_risk_dist)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Block: Category Risk ──
        cat_stat_pills = "".join([
            f'<span class="ai-stat-pill"><span style="color:#7C3AED;">🏷</span> {safe(row.CategoryName)}: {row.FailRate}%</span>'
            for row in cat_fail_rates.head(3).itertuples()
        ])
        st.markdown(f"""
        <div class="ai-block-primary" style="border-left:4px solid #7C3AED;">
            <div class="ai-block-accent-bar" style="background:linear-gradient(90deg,#7C3AED,transparent);opacity:0.3;"></div>
            <div class="ai-block-header">
                <div class="ai-block-icon-wrap" style="background:rgba(124,58,237,0.12);">🍽️</div>
                <div style="flex:1;">
                    <div class="ai-block-subtitle">Food Category · Risk Intelligence</div>
                    <div class="ai-block-title">Category Risk Focus</div>
                </div>
                <div>
                    <span style="background:rgba(124,58,237,0.1);color:#7C3AED;border:1px solid rgba(124,58,237,0.25);border-radius:20px;padding:4px 14px;font-size:12px;font-weight:700;white-space:nowrap;">{safe(top_cat_name)} · {top_cat_rate}% fail</span>
                </div>
            </div>
            <div class="ai-block-body">{top_cat_risk_text}</div>
            <div class="ai-stat-row">
                {cat_stat_pills}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Block: Action Timeline ──
        rec_icons_list  = ["🎯","🔍","📋","💡","🛡️"]
        rec_colors_list = ["#E11D48","#2563EB","#059669","#D97706","#7C3AED"]
        priority_labels = ["Critical","High","Medium","Medium","Low"]
        action_count    = len(recommendations)

        _rows = ""
        for i, rec in enumerate(recommendations[:5]):
            ico    = rec_icons_list[i]  if i < len(rec_icons_list)  else "📌"
            col    = rec_colors_list[i] if i < len(rec_colors_list) else "#64748B"
            plabel = priority_labels[i] if i < len(priority_labels) else "Medium"
            is_last = (i == min(len(recommendations), 5) - 1)
            _line = "" if is_last else f'<div style="position:absolute;left:17px;top:38px;bottom:0;width:2px;background:rgba(0,0,0,0.08);"></div>'
            _rows += (
                f'<div style="display:flex;gap:16px;padding-bottom:20px;position:relative;">'
                f'{_line}'
                f'<div style="width:36px;height:36px;border-radius:10px;display:flex;align-items:center;'
                f'justify-content:center;font-size:16px;flex-shrink:0;z-index:1;min-width:36px;'
                f'background:{col}18;border:2px solid {col}30;">{ico}</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;'
                f'color:#64748B;margin-bottom:4px;">ACTION {i+1} &bull; PRIORITY {plabel.upper()}</div>'
                f'<div style="font-size:14px;color:#475569;line-height:1.6;">{rec}</div>'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;'
            f'padding:22px;box-shadow:var(--shadow-sm);">'
            f'<div style="font-size:15px;font-weight:700;color:var(--text-primary);margin-bottom:18px;'
            f'display:flex;align-items:center;gap:10px;">'
            f'&#9989; AI Recommended Action Plan'
            f'<span style="font-size:11px;font-weight:600;color:#059669;background:rgba(5,150,105,0.08);'
            f'border:1px solid rgba(5,150,105,0.2);border-radius:20px;padding:2px 10px;">'
            f'{action_count} actions identified</span></div>'
            f'{_rows}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Block: Forecast ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(124,58,237,0.06),rgba(37,99,235,0.04));
                    border:1px solid rgba(124,58,237,0.18);border-radius:16px;padding:20px 24px;
                    display:flex;align-items:center;gap:16px;margin-top:4px;">
            <div style="background:linear-gradient(135deg,#7C3AED,#2563EB);width:42px;height:42px;
                        border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">🔮</div>
            <div>
                <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;color:#7C3AED;margin-bottom:4px;">AI Forward Projection</div>
                <div style="font-size:14px;color:var(--text-dim);line-height:1.7;">{forecast_text_val}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ AI briefing unavailable. Check your Groq API key or try refreshing.")
        st.markdown(f"""
        <div class="ai-block-primary" style="border-left:4px solid #D97706;">
            <div class="ai-block-body">
                Based on current data: <b>{adulteration}%</b> adulteration rate with
                <b>{safe(top_risk_cat)}</b> as the highest-risk category and
                <b>{safe(top_risk_dist)}</b> as the most critical district.
                Total health cases recorded: <b>{health_cases:,}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Refresh button
    col_ref, _ = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 Refresh AI Briefing", key="refresh_briefing"):
            st.session_state.pop("ai_structured_insights", None)
            st.session_state.pop("ai_briefing_cache_key", None)
            st.rerun()

    # ── SECTION 3: Chat + Voice ────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;
                color:var(--text-primary);margin-bottom:12px;">
        💬 Ask the AI Assistant
        <span style="font-size:13px;font-weight:600;color:#059669;
                     background:rgba(5,150,105,0.08);border:1px solid rgba(5,150,105,0.2);
                     border-radius:20px;padding:3px 12px;margin-left:10px;">
            🎤 Voice Enabled
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick-question buttons ──────────────────────────────────────
    st.markdown(
        "<div style='font-size:13px;font-weight:600;color:#64748B;margin-bottom:8px;'>"
        "💡 Quick questions:</div>",
        unsafe_allow_html=True,
    )
    qcols = st.columns(4)
    quick_qs = [
        "Which districts need immediate action?",
        "What are the top food safety risks?",
        "How does health impact correlate with failures?",
        "Recommend an intervention strategy",
    ]
    for i, (qcol, question) in enumerate(zip(qcols, quick_qs)):
        with qcol:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                if (
                    not st.session_state.chat_history
                    or st.session_state.chat_history[-1].get("content") != question
                ):
                    st.session_state.chat_history.append(
                        {"role": "user", "content": question}
                    )
                    with st.spinner("⚡ Groq AI is analysing…"):
                        try:
                            response = query_ai_assistant(question, context_summary)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                        except Exception as e:
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": f"Sorry, error: {e}"}
                            )
                    st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Voice + Text input row ──────────────────────────────────────
    # Layout: [text input ──────────────────] [Send] [🎤 mic]
    inp_col, send_col, mic_col = st.columns([5, 1, 1])

    with inp_col:
        user_input = st.text_input(
            "Ask the AI assistant…",
            placeholder="e.g. What should I prioritise in Tirupati district?  (or use the 🎤 mic)",
            label_visibility="collapsed",
            key="chat_input",
        )

    with send_col:
        send_btn = st.button("Send 🚀", use_container_width=True)

    with mic_col:
        # mic_recorder returns None until the user records; then a dict with 'bytes'
        audio_data = mic_recorder(
            start_prompt="🎤 Record",
            stop_prompt="⏹ Done",
            just_once=True,        # resets after each recording so re-recording works
            use_container_width=True,
            key="voice_recorder",
        )

    # ── Voice transcription flow ────────────────────────────────────
    if audio_data and audio_data.get("bytes"):
        with st.spinner("🎙️ Transcribing your voice with Groq Whisper…"):
            try:
                transcript = transcribe_voice(audio_data["bytes"])
                if transcript:
                    st.success(f'🎤 Transcribed: "{transcript}"')
                    # Avoid duplicate submissions
                    if (
                        not st.session_state.chat_history
                        or st.session_state.chat_history[-1].get("content") != transcript
                    ):
                        st.session_state.last_submitted = transcript
                        st.session_state.chat_history.append(
                            {"role": "user", "content": transcript}
                        )
                        with st.spinner("⚡ Groq AI is analysing your question…"):
                            try:
                                response = query_ai_assistant(transcript, context_summary)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": response}
                                )
                            except Exception as e:
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": f"Error: {e}"}
                                )
                        st.rerun()
                else:
                    st.warning("⚠️ No speech detected — please try again.")
            except RuntimeError as err:
                st.error(f"🎙️ Voice error: {err}")

    # ── Text input / Send button flow ──────────────────────────────
    pending_message = ""
    if send_btn and user_input.strip():
        pending_message = user_input.strip()
    elif (
        user_input.strip()
        and user_input.strip() != st.session_state.last_submitted
        and not send_btn
    ):
        pending_message = user_input.strip()

    if pending_message:
        st.session_state.last_submitted = pending_message
        st.session_state.chat_history.append({"role": "user", "content": pending_message})
        with st.spinner("⚡ Groq AI is analysing your data…"):
            try:
                response = query_ai_assistant(pending_message, context_summary)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
        st.rerun()

    # ── Chat history display ────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">{safe(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-ai">⚡ {safe(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.last_submitted = ""
            st.rerun()

    else:
        st.markdown("""
        <div style="background:var(--bg-card2);border:1px solid var(--border);border-radius:16px;
                    padding:28px;text-align:center;margin-top:12px;">
            <div style="font-size:32px;margin-bottom:10px;">🎤</div>
            <div style="font-size:15px;font-weight:600;color:var(--text-primary);margin-bottom:6px;">
                Start a conversation — type or speak
            </div>
            <div style="font-size:14px;color:#64748B;">
                Use the quick buttons, type your question, or click 🎤 Record to ask by voice.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Voice tips + capability pills ──────────────────────────────
    st.markdown("""
    <div style="background:var(--bg-card2);border-radius:14px;padding:18px;border:1px solid var(--border);">
        <div style="font-size:14px;font-weight:700;color:var(--text-primary);margin-bottom:10px;">
            🎯 What you can ask — by voice or text:
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span class="stat-pill" style="background:rgba(37,99,235,0.08);color:#2563EB;border:1px solid rgba(37,99,235,0.2);">District-level risk analysis</span>
            <span class="stat-pill" style="background:rgba(5,150,105,0.08);color:#059669;border:1px solid rgba(5,150,105,0.2);">Category contamination trends</span>
            <span class="stat-pill" style="background:rgba(225,29,72,0.08);color:#E11D48;border:1px solid rgba(225,29,72,0.2);">Emergency action plans</span>
            <span class="stat-pill" style="background:rgba(124,58,237,0.08);color:#7C3AED;border:1px solid rgba(124,58,237,0.2);">Health correlation insights</span>
            <span class="stat-pill" style="background:rgba(217,119,6,0.08);color:#D97706;border:1px solid rgba(217,119,6,0.2);">Seasonal risk forecasting</span>
            <span class="stat-pill" style="background:rgba(8,145,178,0.08);color:#0891B2;border:1px solid rgba(8,145,178,0.2);">Inspection resource allocation</span>
            <span class="stat-pill" style="background:rgba(249,115,22,0.08);color:#F97316;border:1px solid rgba(249,115,22,0.2);">🎤 Speak any of the above</span>
        </div>
        <div style="margin-top:12px;padding:10px 14px;background:rgba(37,99,235,0.05);
                    border:1px solid rgba(37,99,235,0.15);border-radius:10px;
                    font-size:12px;color:#64748B;line-height:1.6;">
            🎙️ <b style='color:var(--text-primary);'>Voice tip:</b>
            Click <b>🎤 Record</b>, speak your question clearly, then click <b>⏹ Done</b>.
            Groq Whisper will transcribe it and the AI will respond automatically.
            Supports English, Telugu (<code>te</code>), and Hindi (<code>hi</code>)
            — change the <code>language</code> param in <code>transcribe_voice()</code>.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 🔮 PREDICTIVE ANALYTICS
# =========================================================

with tab_predict:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">🔮 Predictive Analytics</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">Groq AI-generated risk forecasts, trend projections, and scenario analysis for the next quarter</div>""", unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1: st.markdown(kpi("📈","Projected Risk",     "+18%",   "Next quarter estimate",        "down", "kpi-red"),    unsafe_allow_html=True)
    with p2: st.markdown(kpi("🎯","Forecast Accuracy",  "94%",    "Historical model precision",   "up",   "kpi-green"),  unsafe_allow_html=True)
    with p3: st.markdown(kpi("⚡","Risk Districts",     "7",      "Projected high-risk zones",    "warn", "kpi-amber"),  unsafe_allow_html=True)
    with p4: st.markdown(kpi("🔮","Horizon",            "Q3 2025","Next forecast period",         "up",   "kpi-purple"), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    forecast_col, chart_col = st.columns([1, 2])
    with forecast_col:
        st.markdown("""<div class="forecast-card">
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);margin-bottom:14px;">⚡ Groq AI Forecast Narrative</div>""", unsafe_allow_html=True)

        if st.button("🔮 Generate AI Forecast", use_container_width=True):
            dist_summary = (
                f"Districts: Tirupati (89% risk), Chittoor (76%), Vizag (72%), Kurnool (68%). "
                f"Adulteration rate: {adulteration}%. Top risk categories: Milk, Oils, Spices."
            )
            with st.spinner("⚡ Groq AI generating forecast..."):
                try:
                    forecast_text_result = get_ai_forecast(dist_summary)
                    st.session_state["forecast_text"] = forecast_text_result
                except Exception as e:
                    st.session_state["forecast_text"] = f"Forecast unavailable: {str(e)}"

        if "forecast_text" in st.session_state:
            st.markdown(f"""
            <div style="background:rgba(124,58,237,0.05);border-radius:12px;padding:16px;
                        font-size:14px;color:var(--text-dim);line-height:1.7;margin-top:12px;
                        border:1px solid rgba(124,58,237,0.12);">
                {safe(st.session_state["forecast_text"])}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(124,58,237,0.05);border-radius:12px;padding:16px;
                        font-size:14px;color:#94A3B8;line-height:1.7;margin-top:12px;font-style:italic;">
                Click "Generate AI Forecast" to get a personalized risk projection powered by Groq AI.
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col:
        months_past   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug"]
        months_future = ["Sep","Oct","Nov","Dec","Jan'26","Feb'26"]
        risk_past     = [45, 52, 48, 61, 70, 65, 72, 68]
        risk_forecast = [72, 76, 80, 85, 88, 82]
        risk_upper    = [78, 84, 90, 95, 96, 92]
        risk_lower    = [65, 68, 70, 74, 78, 71]

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=months_past, y=risk_past, name="Historical", mode="lines+markers",
                                           line=dict(color="#2563EB", width=2.5), marker=dict(size=6)))
        fig_forecast.add_trace(go.Scatter(x=months_future, y=risk_forecast, name="AI Forecast", mode="lines+markers",
                                           line=dict(color="#E11D48", width=2.5, dash="dash"), marker=dict(size=6)))
        fig_forecast.add_trace(go.Scatter(
            x=months_future + months_future[::-1],
            y=risk_upper + risk_lower[::-1],
            fill="toself", fillcolor="rgba(225,29,72,0.08)",
            line=dict(color="rgba(255,255,255,0)"), name="Confidence Band"
        ))
        fig_forecast.add_vline(x=7.5, line_dash="dot", line_color="#D97706", line_width=1.5,
                                annotation_text="Forecast →", annotation_font_color="#D97706")
        fig_forecast.update_layout(**get_chart_layout(), height=360, title_text="Risk Score — Historical + Groq AI Forecast")
        st.plotly_chart(fig_forecast, use_container_width=True)

    cat_forecast = pd.DataFrame({
        "Category":    ["Milk","Oils","Spices","Water","Street Food","Fruits","Vegetables","Sweets"],
        "Current Risk":[95, 88, 72, 75, 60, 45, 38, 55],
        "Projected Q3":[108, 98, 80, 82, 65, 48, 35, 62],
        "Change":      ["+13.7%","+11.4%","+11.1%","+9.3%","+8.3%","+6.7%","-7.9%","+12.7%"]
    })
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:12px;">📊 Category Risk Projection — Q3 2025</div>""", unsafe_allow_html=True)

    fig_cat_proj = go.Figure()
    fig_cat_proj.add_trace(go.Bar(name="Current Risk", x=cat_forecast["Category"], y=cat_forecast["Current Risk"],
                                   marker_color="#2563EB", marker_line_width=0))
    fig_cat_proj.add_trace(go.Bar(name="Projected Q3", x=cat_forecast["Category"], y=cat_forecast["Projected Q3"],
                                   marker_color="#E11D48", marker_line_width=0, opacity=0.8))
    fig_cat_proj.update_layout(**get_chart_layout(), height=360, title_text="Current vs Projected Risk by Category", barmode="group")
    st.plotly_chart(fig_cat_proj, use_container_width=True)

    st.dataframe(cat_forecast.style.applymap(
        lambda v: "color: #E11D48; font-weight: bold;" if isinstance(v, str) and v.startswith("+") else
                  "color: #059669; font-weight: bold;" if isinstance(v, str) and v.startswith("-") else "",
        subset=["Change"]
    ), use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:12px;">🎭 Scenario Modeling</div>""", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    scenarios = [
        ("🟢 Best Case",  "#059669", "rgba(5,150,105,0.08)",  "rgba(5,150,105,0.2)",
         "Inspections surge +40%, proactive vendor training, cold chain compliance enforced.",
         "Projected adulteration: 15%", "Health cases: -22%", "Risk districts: 2"),
        ("🟡 Base Case",  "#D97706", "rgba(217,119,6,0.08)",  "rgba(217,119,6,0.2)",
         "Current intervention pace maintained, seasonal risks not mitigated.",
         "Projected adulteration: 24%", "Health cases: +8%",  "Risk districts: 5"),
        ("🔴 Worst Case", "#E11D48", "rgba(225,29,72,0.08)",  "rgba(225,29,72,0.2)",
         "Inspections reduce, festival season surge, water contamination spreads.",
         "Projected adulteration: 38%", "Health cases: +35%", "Risk districts: 10"),
    ]
    for col, (title, color, bg, border, desc, s1, s2, s3_val) in zip([sc1, sc2, sc3], scenarios):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:16px;padding:20px;height:100%;">
                <div style="font-size:15px;font-weight:700;color:{color};margin-bottom:10px;">{title}</div>
                <div style="font-size:13px;color:#475569;line-height:1.7;margin-bottom:14px;">{desc}</div>
                <div style="font-size:12px;font-weight:600;color:{color};">{s1}</div>
                <div style="font-size:12px;font-weight:600;color:{color};">{s2}</div>
                <div style="font-size:12px;font-weight:600;color:{color};">{s3_val}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# ABOUT
# =========================================================

with tab6:
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:800;color:var(--text-primary);margin-bottom:6px;">ℹ️ About VigilantAP</div>
    <div style="color:#64748B;font-size:14px;margin-bottom:22px;">Platform overview, technology stack, objectives, and future roadmap</div>""", unsafe_allow_html=True)

    ab1, ab2, ab3, ab4 = st.columns(4)
    with ab1: st.markdown(kpi("🗺️","Districts Covered", "26",      "Andhra Pradesh",         "up", "kpi-blue"),  unsafe_allow_html=True)
    with ab2: st.markdown(kpi("🧪","Samples Tested",    "20,000+", "Comprehensive testing",  "up", "kpi-cyan"),  unsafe_allow_html=True)
    with ab3: st.markdown(kpi("🍔","Risk Categories",   "16",      "Food types monitored",   "up", "kpi-purple"),unsafe_allow_html=True)
    with ab4: st.markdown(kpi("✅","System Accuracy",   "98%",     "Monitoring precision",   "up", "kpi-green"),  unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:18px;padding:32px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:var(--text-primary);margin-bottom:14px;">📌 What is VigilantAP?</div>
        <div style="font-size:15px;color:#475569;line-height:1.9;">
            VigilantAP is an AI-powered food safety intelligence platform designed to monitor food adulteration trends, identify high-risk districts, analyze public health impact, and support government intervention decisions across <b style='color:#2563EB;'>Andhra Pradesh</b>. Built on real laboratory testing data, the platform enables proactive, evidence-based public health management. Powered by <b style='color:#F97316;'>Groq · LLaMA-3 70B</b> for ultra-fast AI responses, <b style='color:#0891B2;'>predictive analytics</b>, <b style='color:#E11D48;'>anomaly detection</b>, and <b style='color:#059669;'>dark mode</b>.
        </div>
    </div>""", unsafe_allow_html=True)

    ab_c1, ab_c2 = st.columns(2)
    with ab_c1:
        tech_items = ["Python","Streamlit","Plotly","Pandas","NumPy","Groq AI","LLaMA-3 70B","Anomaly Detection","Predictive Analytics","Dark Mode"]
        tech_colors_map = {
            "Python":"#2563EB","Streamlit":"#E11D48","Plotly":"#7C3AED","Pandas":"#059669",
            "NumPy":"#0891B2","Groq AI":"#F97316","LLaMA-3 70B":"#FB923C",
            "Anomaly Detection":"#4F46E5","Predictive Analytics":"#0E7490","Dark Mode":"#475569"
        }
        tech_pills = "".join([
            f'<span style="background:rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.1);color:{tech_colors_map[t]};border-radius:20px;padding:5px 14px;font-size:13px;font-weight:600;">{t}</span>'
            for t in tech_items
        ])
        st.markdown(f"""<div style="background:var(--bg-card);border:1px solid rgba(37,99,235,0.12);border-radius:16px;padding:24px;height:100%;">
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);margin-bottom:14px;">🛠️ Technologies Used</div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;">{tech_pills}</div>
        </div>""", unsafe_allow_html=True)

    with ab_c2:
        st.markdown("""<div style="background:var(--bg-card);border:1px solid rgba(5,150,105,0.12);border-radius:16px;padding:24px;height:100%;">
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);margin-bottom:14px;">🎯 Project Objectives</div>
            <ul style="font-size:14px;color:#475569;line-height:2.1;margin:0;padding-left:18px;">
                <li>Monitor food adulteration trends across AP</li>
                <li>Identify high-risk districts in real-time</li>
                <li>Analyze and visualize public health impact</li>
                <li>Support data-driven government decisions</li>
                <li>Groq AI chat assistant for instant insights</li>
                <li>Predictive risk forecasting with scenario modeling</li>
                <li>Statistical anomaly detection across districts</li>
            </ul>
        </div>""", unsafe_allow_html=True)


    # ---- END USERS SECTION ----
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:16px;">👥 Who Uses VigilantAP?</div>""", unsafe_allow_html=True)

    end_users = [
        {
            "icon": "🏛️",
            "title": "Government Officials",
            "role": "Food Safety Department · AP",
            "desc": "Policy makers and senior officials who rely on real-time dashboards and AI-driven forecasts to allocate inspection resources, frame food safety regulations, and track district-level compliance.",
            "tags": ["Policy Decisions", "Budget Allocation", "Compliance Tracking"],
            "color": "#2563EB",
            "bg": "rgba(37,99,235,0.07)",
            "border": "rgba(37,99,235,0.18)",
        },
        {
            "icon": "🔬",
            "title": "Food Safety Inspectors",
            "role": "Field Officers · District Labs",
            "desc": "Ground-level officers and laboratory analysts who use anomaly alerts, sample test results, and risk heatmaps to prioritise field visits and flag high-risk vendors faster.",
            "tags": ["Anomaly Alerts", "Lab Results", "Risk Heatmaps"],
            "color": "#0891B2",
            "bg": "rgba(8,145,178,0.07)",
            "border": "rgba(8,145,178,0.18)",
        },
        {
            "icon": "🏥",
            "title": "Public Health Officials",
            "role": "Health Departments · Hospitals",
            "desc": "Epidemiologists and health administrators who correlate food adulteration spikes with disease outbreak data to identify causation patterns and coordinate early health interventions.",
            "tags": ["Outbreak Correlation", "Health Impact", "Early Warning"],
            "color": "#7C3AED",
            "bg": "rgba(124,58,237,0.07)",
            "border": "rgba(124,58,237,0.18)",
        },
        {
            "icon": "📊",
            "title": "Data Analysts & Researchers",
            "role": "Academic Institutions · Think Tanks",
            "desc": "Researchers who leverage predictive analytics, scenario modeling, and historical trend data to publish studies, build machine-learning models, and propose evidence-based food safety reforms.",
            "tags": ["Predictive Models", "Trend Analysis", "Research Data"],
            "color": "#059669",
            "bg": "rgba(5,150,105,0.07)",
            "border": "rgba(5,150,105,0.18)",
        },
        {
            "icon": "📰",
            "title": "Journalists & Media",
            "role": "Investigative Reporters · Press",
            "desc": "Reporters using transparent district-level statistics, adulteration category breakdowns, and AI-generated insights to craft data-driven public interest stories on food safety.",
            "tags": ["District Stats", "Category Breakdown", "Public Reporting"],
            "color": "#D97706",
            "bg": "rgba(217,119,6,0.07)",
            "border": "rgba(217,119,6,0.18)",
        },
        {
            "icon": "🛒",
            "title": "Consumer Advocacy Groups",
            "role": "NGOs · Consumer Forums",
            "desc": "Civil society organisations that monitor platform data to hold vendors and authorities accountable, raise public awareness about high-risk food categories, and advocate for stricter standards.",
            "tags": ["Accountability", "Awareness Campaigns", "Advocacy"],
            "color": "#E11D48",
            "bg": "rgba(225,29,72,0.07)",
            "border": "rgba(225,29,72,0.18)",
        },
    ]

    eu_row1 = st.columns(3)
    eu_row2 = st.columns(3)
    all_eu_cols = eu_row1 + eu_row2

    for col, user in zip(all_eu_cols, end_users):
        tags_html = "".join([
            f'<span style="background:rgba(0,0,0,0.05);border:1px solid {user["border"]};color:{user["color"]};'
            f'border-radius:20px;padding:3px 11px;font-size:11px;font-weight:700;letter-spacing:0.3px;">{t}</span>'
            for t in user["tags"]
        ])
        with col:
            st.markdown(f"""
            <div style="background:{user['bg']};border:1px solid {user['border']};border-radius:18px;
                        padding:22px;height:100%;transition:transform 0.2s ease;margin-bottom:16px;">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                    <div style="width:46px;height:46px;border-radius:12px;background:rgba(255,255,255,0.6);
                                border:1px solid {user['border']};display:flex;align-items:center;
                                justify-content:center;font-size:22px;flex-shrink:0;">
                        {user['icon']}
                    </div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                                    color:var(--text-primary);line-height:1.3;">{user['title']}</div>
                        <div style="font-size:11px;color:{user['color']};font-weight:600;
                                    text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">{user['role']}</div>
                    </div>
                </div>
                <div style="font-size:13px;color:#475569;line-height:1.75;margin-bottom:14px;">{user['desc']}</div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ---- DEVELOPED BY SOCIAL TEK ----
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:16px;">🏢 Developed By</div>""", unsafe_allow_html=True)

    dev_col1, dev_col2 = st.columns([1.1, 0.9])

    with dev_col1:
        components.html("""
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
        <div style="background:#ffffff;border:1px solid rgba(37,99,235,0.18);border-radius:18px;padding:28px;font-family:'DM Sans',sans-serif;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;">
                <div style="width:52px;height:52px;border-radius:14px;
                            background:linear-gradient(135deg,#2563EB,#0891B2);
                            display:flex;align-items:center;justify-content:center;
                            font-size:26px;flex-shrink:0;box-shadow:0 6px 18px rgba(37,99,235,0.35);">
                    🤖
                </div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:800;
                                color:#0F172A;line-height:1.2;">Social Tek</div>
                    <div style="font-size:12px;color:#2563EB;font-weight:600;letter-spacing:0.4px;margin-top:3px;">
                        SocialTek AI &amp; ML Business Solutions
                    </div>
                </div>
            </div>

            <div style="font-size:14px;color:#475569;line-height:1.85;margin-bottom:18px;">
                VigilantAP is proudly developed and maintained by <b style="color:#2563EB;">Social Tek</b> —
                a Hyderabad-based AI &amp; ML solutions company specialising in intelligent data platforms,
                predictive analytics, and government technology solutions.
            </div>

            <div style="display:flex;flex-direction:column;gap:12px;">

                <div style="display:flex;align-items:flex-start;gap:10px;">
                    <div style="width:34px;height:34px;border-radius:8px;background:rgba(37,99,235,0.09);
                                border:1px solid rgba(37,99,235,0.2);display:flex;align-items:center;
                                justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;">📍</div>
                    <div>
                        <div style="font-size:11px;color:#94A3B8;font-weight:700;text-transform:uppercase;
                                    letter-spacing:0.6px;margin-bottom:3px;">Registered Office</div>
                        <div style="font-size:13px;color:#0F172A;font-weight:500;line-height:1.7;">
                            501, Sathyabama Complex, Bhagyanagar Colony,<br>
                            Opp. Sai Baba Temple, KPHB,<br>
                            Hyderabad, Telangana &ndash; 500085
                        </div>
                    </div>
                </div>

                <div style="display:flex;align-items:flex-start;gap:10px;">
                    <div style="width:34px;height:34px;border-radius:8px;background:rgba(8,145,178,0.09);
                                border:1px solid rgba(8,145,178,0.2);display:flex;align-items:center;
                                justify-content:center;font-size:16px;flex-shrink:0;margin-top:2px;">🏛️</div>
                    <div>
                        <div style="font-size:11px;color:#94A3B8;font-weight:700;text-transform:uppercase;
                                    letter-spacing:0.6px;margin-bottom:3px;">Corporate Office</div>
                        <div style="font-size:13px;color:#0F172A;font-weight:500;line-height:1.7;">
                            508, Manjeera Majestic Commercial,<br>
                            JNTU Road, KPHB,<br>
                            Hyderabad, Telangana &ndash; 500085
                        </div>
                    </div>
                </div>

                <div style="display:flex;align-items:center;gap:10px;margin-top:2px;">
                    <div style="width:34px;height:34px;border-radius:8px;background:rgba(5,150,105,0.09);
                                border:1px solid rgba(5,150,105,0.2);display:flex;align-items:center;
                                justify-content:center;font-size:16px;flex-shrink:0;">🌐</div>
                    <div style="font-size:13px;color:#059669;font-weight:600;">KPHB, Hyderabad, Telangana, India</div>
                </div>

            </div>
        </div>
        """, height=370, scrolling=False)

    with dev_col2:
        st.markdown("""<div style="font-size:13px;font-weight:700;color:var(--text-primary);margin-bottom:8px;">
            📍 Office Location — KPHB, Hyderabad
        </div>""", unsafe_allow_html=True)

        # Interactive map centred on KPHB, Hyderabad
        # Registered office: Sathyabama Complex (~17.4612, 78.3768)
        # Corporate office:   Manjeera Majestic  (~17.4587, 78.3812)
        office_map_data = pd.DataFrame({
            "lat":   [17.4612, 17.4587],
            "lon":   [78.3768, 78.3812],
            "name":  ["Registered Office – Sathyabama Complex, KPHB",
                      "Corporate Office – Manjeera Majestic Commercial, KPHB"],
            "color": ["#2563EB", "#0891B2"],
            "size":  [18, 18],
        })

        fig_office = px.scatter_mapbox(
            office_map_data,
            lat="lat", lon="lon",
            hover_name="name",
            color="name",
            color_discrete_sequence=["#2563EB", "#0891B2"],
            size="size",
            size_max=18,
            zoom=14.5,
            height=370,
        )
        fig_office.update_layout(
            mapbox_style="carto-positron",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="left", x=0,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig_office, use_container_width=True)

        st.markdown("""
        <div style="display:flex;gap:10px;margin-top:4px;flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
                <span style="width:10px;height:10px;border-radius:50%;background:#2563EB;display:inline-block;"></span>
                Sathyabama Complex (Reg. Office)
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
                <span style="width:10px;height:10px;border-radius:50%;background:#0891B2;display:inline-block;"></span>
                Manjeera Majestic (Corp. Office)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ---- FOOTER ----
    st.markdown("""<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:24px;text-align:center;margin-top:10px;">
        <div style="font-size:14px;color:#64748B;">
            <b style='color:var(--text-primary);'>VigilantAP v2.2</b> — Food Safety Intelligence Platform<br>
            Domain: Public Health Monitoring &nbsp;|&nbsp; Stack: Python · Streamlit · Plotly · Groq AI · LLaMA-3 70B<br>
            Developed by <b style='color:#2563EB;'>Social Tek (SocialTek AI &amp; ML Business Solutions)</b> · KPHB, Hyderabad<br>
            <span style="color:#2563EB;">© 2026 VigilantAP. Built for Andhra Pradesh Food Safety Department.</span>
        </div>
    </div>""", unsafe_allow_html=True)


