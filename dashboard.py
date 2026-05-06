"""
Workforce Intelligence System — Executive Dashboard
Light, premium, boardroom-ready UI for HR leaders and executives.
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import math

# ─── Design Tokens ──────────────────────────────────────────────────
C = {
    "bg": "#F5F5F7", "white": "#FFFFFF", "card": "#FFFFFF",
    "text": "#1D1D1F", "text2": "#6E6E73", "text3": "#86868B",
    "border": "#D2D2D7", "border_light": "#E8E8ED",
    "blue": "#0071E3", "blue2": "#2997FF", "blue_bg": "#EBF5FF",
    "green": "#34C759", "green_dark": "#248A3D", "green_bg": "#E8FAF0",
    "red": "#FF3B30", "red_dark": "#D70015", "red_bg": "#FFF0EF",
    "orange": "#FF9500", "orange_dark": "#C93400", "orange_bg": "#FFF8EC",
    "purple": "#AF52DE", "purple_bg": "#F5EEFA",
    "teal": "#5AC8FA", "teal_bg": "#EBF8FF",
    "indigo": "#5856D6",
    "gradient_blue": "linear-gradient(135deg, #0071E3, #2997FF)",
    "gradient_green": "linear-gradient(135deg, #248A3D, #34C759)",
    "gradient_red": "linear-gradient(135deg, #D70015, #FF3B30)",
    "gradient_orange": "linear-gradient(135deg, #C93400, #FF9500)",
    "gradient_purple": "linear-gradient(135deg, #7B2CBF, #AF52DE)",
}

st.set_page_config(page_title="Workforce Intelligence", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")

DATA_DIR = Path(__file__).parent / "data"

# Industry registry for the sidebar selector + data routing.
# The dashboard calls set_industry() whenever the sidebar dropdown changes
# so every downstream tool (forecast_tools, talent_tools, brain) sees the
# right profile. The CSVs it loads come from the profile's data_subdir.
from industries import REGISTRY, set_industry as _set_industry

# ─── CSS ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { background: #F5F5F7 !important; }
    .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li, .stSelectbox label,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif !important;
    }
    .block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important; border-right: 1px solid #E8E8ED !important;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #F5F5F7 !important; border: 1px solid #D2D2D7 !important;
        border-radius: 10px !important; color: #1D1D1F !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span { color: #6E6E73 !important; }

    /* Typography */
    .stMarkdown h1 { font-weight: 800 !important; font-size: 2.2rem !important;
        color: #1D1D1F !important; letter-spacing: -0.04em !important; line-height: 1.1 !important; }
    .stMarkdown h2 { font-weight: 700 !important; font-size: 1.4rem !important;
        color: #1D1D1F !important; letter-spacing: -0.03em !important; margin-top: 1.5rem !important; }
    .stMarkdown h3 { font-weight: 600 !important; font-size: 1.05rem !important;
        color: #1D1D1F !important; letter-spacing: -0.02em !important; }
    .stMarkdown p, .stMarkdown li { color: #6E6E73 !important; line-height: 1.6 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #EFEFF4 !important;
        border-radius: 10px !important; padding: 3px !important; border: none !important; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: #6E6E73 !important;
        font-weight: 500 !important; font-size: 0.85rem !important; padding: 7px 18px !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #FFFFFF !important;
        color: #1D1D1F !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Buttons */
    .stButton > button { background: #0071E3 !important; color: #fff !important;
        border: none !important; border-radius: 10px !important; font-weight: 600 !important;
        padding: 10px 28px !important; transition: all 0.15s !important; }
    .stButton > button:hover { background: #0077ED !important; box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important; }

    /* Selectbox */
    .stSelectbox > div > div { background: #fff !important; border: 1px solid #D2D2D7 !important;
        border-radius: 10px !important; color: #1D1D1F !important; }
    .stSelectbox label { color: #86868B !important; font-weight: 500 !important;
        font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }

    /* Textarea */
    .stTextArea textarea { background: #FAFAFA !important; border: 1px solid #E8E8ED !important;
        border-radius: 10px !important; color: #1D1D1F !important;
        font-family: 'SF Mono', 'Fira Code', monospace !important; font-size: 0.8rem !important; }

    /* Hide branding — but keep the sidebar collapse/expand chevron clickable.
       Streamlit renders [data-testid="collapsedControl"] inside the header,
       so a blanket `header { visibility: hidden }` also hides the chevron. */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    header [data-testid="stToolbar"] { visibility: hidden !important; }
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
    }
    /* Keep the sidebar itself visible/expanded by default. Streamlit
       persists the collapsed/expanded state per browser tab in a cookie,
       which can leave the sidebar invisible after a server restart. */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: 0 !important;
        transform: none !important;
    }
    hr { border-color: #E8E8ED !important; }

    /* ── Cards ── */
    .hero-card { background: #fff; border-radius: 20px; padding: 28px 32px; position: relative;
        overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.04); }
    .hero-card .hero-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 10px; }
    .hero-card .hero-value { font-size: 2.8rem; font-weight: 800; letter-spacing: -0.04em;
        line-height: 1; margin-bottom: 6px; color: #1D1D1F; }
    .hero-card .hero-sub { font-size: 0.82rem; color: #86868B; font-weight: 400; }
    .hero-card .hero-accent { position: absolute; top: 0; right: 0; width: 120px; height: 120px;
        border-radius: 0 20px 0 60px; opacity: 0.12; }

    /* Section card */
    .section-card { background: #fff; border-radius: 16px; padding: 24px 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.04); margin-bottom: 16px; }

    /* Entity tags */
    .etag { display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem;
        font-weight: 500; margin: 3px 4px; }
    .etag-skill { background: #EBF5FF; color: #0071E3; }
    .etag-cert { background: #E8FAF0; color: #248A3D; }
    .etag-degree { background: #F5EEFA; color: #7B2CBF; }
    .etag-employer { background: #FFF8EC; color: #C93400; }
    .etag-years { background: #EBF8FF; color: #0077ED; }

    /* Risk badges */
    .rb { display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.72rem;
        font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .rb-critical { background: #FFF0EF; color: #D70015; }
    .rb-high { background: #FFF8EC; color: #C93400; }
    .rb-medium { background: #FFFAEB; color: #866200; }
    .rb-low { background: #E8FAF0; color: #248A3D; }

    /* Match rows */
    .mrow { background: #fff; border: 1px solid #E8E8ED; border-radius: 14px; padding: 16px 20px;
        margin-bottom: 8px; display: flex; align-items: center; gap: 16px;
        transition: box-shadow 0.15s; }
    .mrow:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
    .mrow-rank { font-size: 0.95rem; font-weight: 700; color: #86868B; min-width: 28px; }
    .mrow-name { font-weight: 600; color: #1D1D1F; font-size: 0.95rem; }
    .mrow-tier { font-size: 0.78rem; color: #86868B; margin-top: 1px; }
    .mrow-score { margin-left: auto; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.03em; }

    /* Dept rows */
    .drow { background: #fff; border: 1px solid #E8E8ED; border-radius: 14px; padding: 16px 20px;
        margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; }
    .drow:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
    .drow-name { font-weight: 600; color: #1D1D1F; font-size: 0.95rem; }
    .drow-meta { font-size: 0.78rem; color: #86868B; margin-top: 2px; }

    /* Arch cards */
    .acard { background: #fff; border: 1px solid #E8E8ED; border-radius: 20px; padding: 32px;
        text-align: center; transition: box-shadow 0.15s; }
    .acard:hover { box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
    .acard-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: #86868B; margin-bottom: 12px; }
    .acard-title { font-size: 1.25rem; font-weight: 800; color: #1D1D1F;
        letter-spacing: -0.03em; margin-bottom: 8px; }
    .acard-desc { font-size: 0.82rem; color: #6E6E73; line-height: 1.5; }

    /* ── Magic UI-Inspired Enhancements ── */

    /* 1. Staggered slide-up animations for list rows */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .mrow, .drow { opacity: 0; animation: slideUp 0.4s ease forwards; }
    .mrow:nth-child(1), .drow:nth-child(1) { animation-delay: 0.04s; }
    .mrow:nth-child(2), .drow:nth-child(2) { animation-delay: 0.08s; }
    .mrow:nth-child(3), .drow:nth-child(3) { animation-delay: 0.12s; }
    .mrow:nth-child(4), .drow:nth-child(4) { animation-delay: 0.16s; }
    .mrow:nth-child(5), .drow:nth-child(5) { animation-delay: 0.20s; }
    .mrow:nth-child(6), .drow:nth-child(6) { animation-delay: 0.24s; }
    .mrow:nth-child(7), .drow:nth-child(7) { animation-delay: 0.28s; }
    .mrow:nth-child(8), .drow:nth-child(8) { animation-delay: 0.32s; }

    /* Hero cards also slide up */
    .hero-card { opacity: 0; animation: slideUp 0.5s ease forwards; }

    /* Architecture cards fade in */
    .acard { opacity: 0; animation: slideUp 0.6s ease forwards; animation-delay: 0.1s; }

    /* Entity tags pop in */
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.85); }
        to { opacity: 1; transform: scale(1); }
    }
    .etag { opacity: 0; animation: popIn 0.3s ease forwards; }
    .etag:nth-child(1) { animation-delay: 0.03s; }
    .etag:nth-child(2) { animation-delay: 0.06s; }
    .etag:nth-child(3) { animation-delay: 0.09s; }
    .etag:nth-child(4) { animation-delay: 0.12s; }
    .etag:nth-child(5) { animation-delay: 0.15s; }
    .etag:nth-child(6) { animation-delay: 0.18s; }

    /* 2. Gradient animated text for page titles */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .gradient-title {
        background: linear-gradient(135deg, #0071E3, #AF52DE, #5856D6, #0071E3);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease infinite;
        display: inline-block;
    }

    /* 3. Shimmer loading placeholder */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    .shimmer-card {
        background: linear-gradient(90deg, #F0F0F5 25%, #E8E8ED 37%, #F0F0F5 63%);
        background-size: 200% 100%;
        animation: shimmer 1.8s ease-in-out infinite;
        border-radius: 16px;
        min-height: 120px;
    }
    .shimmer-line {
        background: linear-gradient(90deg, #F0F0F5 25%, #E8E8ED 37%, #F0F0F5 63%);
        background-size: 200% 100%;
        animation: shimmer 1.8s ease-in-out infinite;
        border-radius: 8px;
        height: 16px;
        margin-bottom: 10px;
    }

    /* 4. Number counting animation for hero values */
    @keyframes countFadeIn {
        from { opacity: 0; transform: translateY(10px); filter: blur(4px); }
        to { opacity: 1; transform: translateY(0); filter: blur(0); }
    }
    .hero-card .hero-value {
        animation: countFadeIn 0.7s ease-out forwards;
        animation-delay: 0.2s;
        opacity: 0;
    }

    /* 5. Marquee ticker */
    @keyframes marqueeScroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .marquee-container {
        overflow: hidden;
        background: linear-gradient(135deg, #1D1D1F 0%, #2D2D35 100%);
        border-radius: 14px;
        padding: 14px 0;
        margin-bottom: 20px;
        position: relative;
    }
    .marquee-container::before, .marquee-container::after {
        content: '';
        position: absolute; top: 0; bottom: 0; width: 60px; z-index: 2;
    }
    .marquee-container::before {
        left: 0;
        background: linear-gradient(to right, #1D1D1F, transparent);
    }
    .marquee-container::after {
        right: 0;
        background: linear-gradient(to left, #2D2D35, transparent);
    }
    .marquee-track {
        display: flex;
        width: max-content;
        animation: marqueeScroll 30s linear infinite;
    }
    .marquee-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0 28px;
        white-space: nowrap;
        font-size: 0.82rem;
        font-weight: 500;
        color: #E8E8ED;
        font-family: 'Inter', sans-serif;
    }
    .marquee-dot {
        width: 6px; height: 6px; border-radius: 50%;
        display: inline-block; flex-shrink: 0;
    }
    .marquee-sep {
        color: #555; padding: 0 6px; font-size: 0.7rem;
    }

    /* 6. Glowing border for critical risk cards */
    @keyframes borderGlow {
        0%, 100% { box-shadow: 0 0 8px rgba(255,59,48,0.15), 0 1px 3px rgba(0,0,0,0.04); }
        50% { box-shadow: 0 0 20px rgba(255,59,48,0.25), 0 0 40px rgba(255,59,48,0.08); }
    }
    .drow-critical {
        border: 1px solid rgba(255,59,48,0.3) !important;
        animation: borderGlow 2.5s ease-in-out infinite;
    }
    .drow-high {
        border: 1px solid rgba(255,149,0,0.25) !important;
    }

    /* 7. Dot pattern background for overview */
    .dot-bg {
        background-image: radial-gradient(circle, #D2D2D7 1px, transparent 1px);
        background-size: 24px 24px;
        background-position: 0 0;
        padding: 8px 0;
    }

    /* 8. SVG progress ring styles */
    .ring-container {
        display: inline-flex; align-items: center; gap: 10px;
    }
    .ring-container svg { flex-shrink: 0; }

    /* ── Skill-Gap Analyzer panels ── */

    /* Recommendation tier badges */
    .tier-badge {
        display: inline-block; padding: 6px 18px; border-radius: 22px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tier-STRONG_HIRE { background: linear-gradient(135deg, #248A3D, #34C759); color: #fff; }
    .tier-HIRE { background: linear-gradient(135deg, #0071E3, #2997FF); color: #fff; }
    .tier-INTERVIEW { background: linear-gradient(135deg, #AF52DE, #5856D6); color: #fff; }
    .tier-CONDITIONAL { background: linear-gradient(135deg, #FF9500, #C93400); color: #fff; }
    .tier-DO_NOT_ADVANCE { background: linear-gradient(135deg, #D70015, #FF3B30); color: #fff; }

    /* Criticality badges for JD requirements */
    .crit-badge {
        display: inline-block; padding: 2px 9px; border-radius: 10px;
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .crit-CRITICAL { background: #FFF0EF; color: #D70015; }
    .crit-PREFERRED { background: #FFF8EC; color: #C93400; }
    .crit-STANDARD { background: #EBF5FF; color: #0071E3; }
    .crit-NICE_TO_HAVE { background: #F0F0F5; color: #86868B; }

    /* Gap item row */
    .gap-item {
        background: #fff; border: 1px solid #E8E8ED; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 8px;
    }
    .gap-item-head {
        display: flex; align-items: center; gap: 10px;
        font-weight: 600; color: #1D1D1F; font-size: 0.9rem;
    }
    .gap-item-head .gap-sim {
        margin-left: auto; font-weight: 700; font-size: 0.85rem;
    }
    .gap-item-evidence {
        margin-top: 8px; padding: 8px 12px; border-left: 2px solid #0071E3;
        background: #FAFAFA; border-radius: 0 8px 8px 0;
        font-size: 0.82rem; color: #6E6E73; font-style: italic;
        line-height: 1.5;
    }
    .gap-item-action {
        margin-top: 6px; font-size: 0.78rem; color: #86868B; line-height: 1.5;
    }
    .gap-item-action strong { color: #1D1D1F; font-weight: 600; }
    .gap-item-evidence-meta {
        margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap;
        font-size: 0.72rem;
    }
    .prof-high { color: #248A3D; font-weight: 600; }
    .prof-medium { color: #0071E3; font-weight: 600; }
    .prof-low { color: #86868B; font-weight: 500; }

    /* Coaching path row */
    .coach-row {
        background: linear-gradient(135deg, #EBF5FF 0%, #F5EEFA 100%);
        border: 1px solid #D4E6FC; border-radius: 12px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    .coach-row-title {
        font-weight: 600; color: #1D1D1F; font-size: 0.92rem; margin-bottom: 4px;
    }
    .coach-row-delta {
        font-size: 0.85rem; color: #6E6E73;
    }
    .coach-row-delta .delta-up {
        color: #248A3D; font-weight: 700;
    }

    /* Fit component bar */
    .fit-bar-row { margin-bottom: 10px; }
    .fit-bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.78rem; color: #6E6E73; margin-bottom: 4px;
    }
    .fit-bar-label strong { color: #1D1D1F; font-weight: 600; }
    .fit-bar-track {
        height: 8px; background: #F0F0F5; border-radius: 6px; overflow: hidden;
    }
    .fit-bar-fill {
        height: 100%; border-radius: 6px;
        transition: width 0.8s ease;
    }

    /* Briefing paragraph */
    .briefing-block {
        padding: 14px 18px; border-radius: 12px; margin-bottom: 10px;
        font-size: 0.88rem; line-height: 1.6;
    }
    .briefing-verdict { background: #EBF5FF; border-left: 3px solid #0071E3; color: #1D1D1F; }
    .briefing-strengths { background: #E8FAF0; border-left: 3px solid #34C759; color: #1D1D1F; }
    .briefing-concerns { background: #FFF8EC; border-left: 3px solid #FF9500; color: #1D1D1F; }

    /* ── Agent Memo (Boardroom Brief) ── */
    .agent-memo {
        background: #fff; border: 1px solid #E8E8ED; border-radius: 16px;
        padding: 28px 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 14px;
    }
    .agent-memo .memo-headline {
        font-size: 1.4rem; font-weight: 800; color: #1D1D1F;
        letter-spacing: -0.03em; margin-bottom: 6px; line-height: 1.3;
    }
    .agent-memo .memo-verdict-row {
        display: flex; align-items: center; gap: 10px; margin: 8px 0 18px 0;
    }
    .memo-verdict-badge {
        display: inline-block; padding: 6px 14px; border-radius: 18px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .memo-rule-badge {
        display: inline-block; padding: 4px 10px; border-radius: 10px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
        background: #F0F0F5; color: #6E6E73; font-family: 'SF Mono', monospace;
    }
    .memo-section {
        padding: 14px 18px; margin: 10px 0; border-radius: 12px;
        background: #FAFAFA; border-left: 3px solid #0071E3;
    }
    .memo-section-title {
        font-weight: 700; color: #1D1D1F; font-size: 0.95rem;
        margin-bottom: 8px; letter-spacing: -0.01em;
    }
    .memo-section-body {
        color: #1D1D1F; font-size: 0.88rem; line-height: 1.6;
    }
    .memo-section-body strong { color: #0071E3; font-weight: 600; }
    .memo-section-a { border-left-color: #0071E3; background: #F7FBFF; }
    .memo-section-b { border-left-color: #34C759; background: #F6FDF8; }
    .memo-section-c { border-left-color: #AF52DE; background: #FBF8FD; }
    .memo-footer {
        margin-top: 16px; padding-top: 14px; border-top: 1px solid #F0F0F5;
        display: flex; flex-wrap: wrap; gap: 14px;
        font-size: 0.72rem; color: #86868B;
    }
    .memo-footer-item {
        display: inline-flex; align-items: center; gap: 4px;
    }
    .memo-footer-item strong { color: #1D1D1F; }

    /* Engine pill selector */
    .engine-pill-active {
        background: linear-gradient(135deg, #0071E3, #AF52DE); color: #fff;
        padding: 6px 14px; border-radius: 20px; font-weight: 700;
        font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .engine-pill-inactive {
        background: #F0F0F5; color: #6E6E73;
        padding: 6px 14px; border-radius: 20px; font-weight: 500;
        font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
    }

    /* Execution trace */
    .trace-row {
        display: flex; justify-content: space-between; padding: 6px 12px;
        border-bottom: 1px solid #F0F0F5; font-size: 0.78rem;
        font-family: 'SF Mono', monospace;
    }
    .trace-row:last-child { border-bottom: none; }
    .trace-tool { color: #1D1D1F; font-weight: 500; }
    .trace-ms { color: #86868B; }

    </style>""", unsafe_allow_html=True)


# ─── Data & Models ──────────────────────────────────────────────────
def _industry_data_dir(industry_key: str) -> Path:
    """Resolve the on-disk directory holding `industry_key`'s workforce CSVs.

    Energy keeps its CSVs at `data/` root (data_subdir=""). New verticals
    (Finance, Healthcare, ...) live under `data/<subdir>/`.
    """
    profile = REGISTRY.get(industry_key)
    if profile is None or not profile.data_subdir:
        return DATA_DIR
    return DATA_DIR / profile.data_subdir


@st.cache_data
def load_data(industry_key: str = "energy"):
    """Load per-industry workforce CSVs. Cached on industry_key so toggling
    the sidebar selector triggers a clean reload for the new vertical."""
    ddir = _industry_data_dir(industry_key)
    return (pd.read_csv(ddir / "employees.csv"),
            pd.read_csv(ddir / "monthly_department.csv"),
            pd.read_csv(ddir / "individual_monthly.csv"))

@st.cache_resource
def load_ner():
    """Load the routed NER engine lazily.

    Uses :class:`models.ner_model.RoutedNEREngine`, which wraps the legacy
    DistilBERT + GLiNER ensemble behind ``models/per_class_router.py``. The
    router enforces per-class hard ownership (legacy 5 → v6, TOOL →
    designated TOOL extractor, INDUSTRY/LOCATION/PROJECT/SOFT_SKILL → Gate-2
    winners) and overlays high-precision gazetteer additions.

    Output contract is unchanged: ``extract_entities(text) → Dict[str,
    List[str]]`` covering all 10 production entity types
    (``per_class_router.ALL_TYPES``). The dashboard's existing display code
    iterates 10 types so this is a drop-in replacement.

    Fallback behavior matches the prior implementation: if DistilBERT
    weights are missing, the wrapper falls back to GLiNER-only inside the
    ensemble, and the gazetteer overlay still loads independently.
    """
    try:
        from models.ner_model import RoutedNEREngine
        from tools.talent_tools import set_ner_engine
        ner = RoutedNEREngine()
        ner.load()
        set_ner_engine(ner)
        return ner
    except Exception as e:
        st.error(f"NER load error: {e}")
        return None

@st.cache_resource
def load_sbert():
    """Load SBERT model lazily — only when matching is needed."""
    try:
        from models.sbert_matcher import SBERTMatcher
        from tools.talent_tools import set_sbert_matcher
        sbert = SBERTMatcher()
        set_sbert_matcher(sbert)
        return sbert
    except Exception as e:
        st.error(f"SBERT load error: {e}")
        return None

@st.cache_resource
def load_forecast():
    """Load Bi-LSTM model lazily — only when forecasting page is used."""
    try:
        from models.bilstm_model import ForecastingEngine
        forecast = ForecastingEngine()
        try: forecast.load()
        except Exception: forecast.build_model()
        return forecast
    except Exception as e:
        st.error(f"Forecast load error: {e}")
        return None


# ─── Plotly Styling ─────────────────────────────────────────────────
def style_fig(fig, h=420):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#6E6E73", size=12),
        title=dict(font=dict(size=14, color="#1D1D1F"), x=0, xanchor="left"),
        xaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#E8E8ED"),
        yaxis=dict(gridcolor="#F0F0F0", zerolinecolor="#E8E8ED"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#6E6E73"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=48, b=20),
        hoverlabel=dict(bgcolor="#fff", font_size=12, font_color="#1D1D1F",
                        bordercolor="#D2D2D7"),
        height=h,
    )
    return fig

def score_color(s):
    if s >= 0.80: return C["green_dark"]
    if s >= 0.65: return C["blue"]
    if s >= 0.50: return C["orange"]
    return C["red"]

def rbadge(level):
    cls = {"CRITICAL":"rb-critical","HIGH":"rb-high","MEDIUM":"rb-medium","LOW":"rb-low"}.get(level,"rb-low")
    return f'<span class="rb {cls}">{level}</span>'

def _svg_ring(pct, color, size=36):
    """Generate an SVG progress ring. pct in [0,1], color is hex, size in px."""
    r = (size - 4) / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - min(max(pct, 0), 1))
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg)">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#F0F0F0" stroke-width="3.5"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="3.5" '
        f'stroke-dasharray="{circ}" stroke-dashoffset="{offset}" stroke-linecap="round"'
        f' style="transition: stroke-dashoffset 0.8s ease"/>'
        f'</svg>'
    )


def _tier_badge(tier):
    label = tier.replace("_", " ")
    return f'<span class="tier-badge tier-{tier}">{label}</span>'


def _crit_badge(level):
    label = level.replace("_", " ")
    return f'<span class="crit-badge crit-{level}">{label}</span>'


def _fit_color(score):
    if score >= 85: return C["green_dark"]
    if score >= 70: return C["blue"]
    if score >= 55: return C["purple"]
    if score >= 40: return C["orange"]
    return C["red"]


# Tier cutoffs under three policy families.
#   "standard"   matches tools.talent_tools._fit_to_tier (current production).
#   "stricter"   shifts every threshold up by 10 points — used by orgs with
#                strong "raise the bar" hiring cultures (e.g., FAANG ICs,
#                regulated safety roles where false-positive hires are costly).
#   "looser"     shifts thresholds down by 10 — used in high-volume, fast-fill
#                scenarios where false-negative is expensive.
# The tuples are (STRONG_HIRE_min, HIRE_min, INTERVIEW_min, CONDITIONAL_min).
# Below CONDITIONAL_min → DO_NOT_ADVANCE.
TIER_CUTOFF_POLICIES = {
    "standard": {"label": "Standard",
                  "thresholds": (85, 70, 55, 40),
                  "blurb": "Production cutoffs (matches tools/talent_tools.py)"},
    "stricter": {"label": "Stricter",
                  "thresholds": (90, 80, 65, 50),
                  "blurb": "+10 points everywhere — raise-the-bar policy"},
    "looser":   {"label": "Looser",
                  "thresholds": (80, 60, 45, 30),
                  "blurb": "−10 points everywhere — high-volume sourcing policy"},
}


def _score_to_tier(score, policy="standard"):
    """Map a fit score to the recommendation tier under the named policy."""
    th = TIER_CUTOFF_POLICIES[policy]["thresholds"]
    if score >= th[0]: return "STRONG_HIRE"
    if score >= th[1]: return "HIRE"
    if score >= th[2]: return "INTERVIEW"
    if score >= th[3]: return "CONDITIONAL"
    return "DO_NOT_ADVANCE"


def _confidence_to_band_pp(confidence_score):
    """Derive a ±pp band on the fit score from the analyzer's confidence
    score in [0, 1]. Higher confidence → tighter band. Caps at ±2pp (high
    confidence) and ±15pp (low). Used as an honest signal of how much
    weight choices and noise can move the composite score."""
    try:
        c = float(confidence_score or 0.0)
    except (TypeError, ValueError):
        c = 0.5
    c = max(0.0, min(1.0, c))
    return round(2 + (1 - c) * 13, 1)


def _fit_bar(label, value, color):
    pct = max(0, min(100, value))
    return (
        f'<div class="fit-bar-row">'
        f'  <div class="fit-bar-label"><span>{label}</span>'
        f'<strong>{value:.0f}</strong></div>'
        f'  <div class="fit-bar-track">'
        f'    <div class="fit-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'  </div>'
        f'</div>'
    )


def _render_gap_item(item, is_missing=False):
    """Render one assignment from gaps_by_type[etype][state]."""
    required = item.get("required", "?")
    closest = item.get("closest")
    sim = item.get("similarity", 0.0)
    crit = item.get("criticality", "STANDARD")
    evidence = item.get("candidate_evidence") or {}
    action = item.get("action_suggestion")
    question = item.get("interview_question")
    canonical = item.get("canonical")

    sim_color = C["green_dark"] if sim >= 0.75 else C["orange"] if sim >= 0.50 else C["text3"]
    closest_html = (
        f' <span style="color:#86868B;font-weight:400;font-size:0.82rem">'
        f'&rarr; {closest}</span>'
        if closest and not is_missing else ""
    )
    canon_html = (
        f' <span style="color:#AF52DE;font-size:0.72rem;font-weight:500">'
        f'[{canonical}]</span>'
        if canonical else ""
    )
    sim_html = (
        f'<span class="gap-sim" style="color:{sim_color}">{sim:.2f}</span>'
        if not is_missing else ""
    )

    evidence_html = ""
    if evidence.get("sentence"):
        sent = evidence["sentence"]
        if len(sent) > 180:
            sent = sent[:180] + "…"
        evidence_html += f'<div class="gap-item-evidence">&ldquo;{sent}&rdquo;</div>'
    meta_parts = []
    if evidence.get("proficiency"):
        prof = evidence["proficiency"]
        meta_parts.append(
            f'<span class="prof-{prof}">proficiency: {prof}</span>'
        )
    if evidence.get("mention_count", 0) >= 2:
        meta_parts.append(
            f'<span style="color:#86868B">×{evidence["mention_count"]} mentions</span>'
        )
    if evidence.get("quantifiers"):
        q_str = ", ".join(evidence["quantifiers"][:3])
        meta_parts.append(
            f'<span style="color:#248A3D;font-weight:600">quantified: {q_str}</span>'
        )
    if meta_parts:
        evidence_html += (
            '<div class="gap-item-evidence-meta">' + " · ".join(meta_parts) + '</div>'
        )

    action_html = ""
    if is_missing and action:
        action_html += f'<div class="gap-item-action"><strong>Next step:</strong> {action}</div>'
    if is_missing and question:
        action_html += f'<div class="gap-item-action"><strong>Ask:</strong> {question}</div>'

    return (
        f'<div class="gap-item">'
        f'  <div class="gap-item-head">'
        f'    {_crit_badge(crit)}'
        f'    <span>{required}{canon_html}</span>'
        f'    {closest_html}'
        f'    {sim_html}'
        f'  </div>'
        f'  {evidence_html}'
        f'  {action_html}'
        f'</div>'
    )


DEPT_COLORS = {
    "Operations": "#0071E3", "Engineering": "#AF52DE", "Maintenance": "#FF9500",
    "HSE": "#34C759", "Projects": "#5AC8FA", "Commercial": "#FF3B30",
    "IT": "#5856D6", "HR": "#FF2D55",
}


# ─── OVERVIEW PAGE ──────────────────────────────────────────────────
def render_overview(employees, monthly, individual):
    # Dot pattern background wrapper
    st.markdown('<div class="dot-bg">', unsafe_allow_html=True)

    latest = monthly["month"].max()
    cur = monthly[monthly["month"] == latest]
    hc = int(cur["headcount"].sum())
    depts = len(cur["department"].unique())
    sat = cur["avg_satisfaction"].mean()
    att = individual["attrition"].mean()
    prev = monthly[monthly["month"] == latest - 1]
    hc_prev = int(prev["headcount"].sum()) if len(prev) else hc
    hc_delta = hc - hc_prev

    # Hero metrics
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Total Headcount</div>
            <div class="hero-value">{hc:,}</div>
            <div class="hero-sub">{"+" if hc_delta>=0 else ""}{hc_delta} from last month</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Departments</div>
            <div class="hero-value">{depts}</div>
            <div class="hero-sub">Active divisions</div>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_green']}"></div>
            <div class="hero-label" style="color:{C['green_dark']}">Avg Satisfaction</div>
            <div class="hero-value">{sat:.1f}<span style="font-size:1.2rem;color:#86868B"> / 5</span></div>
            <div class="hero-sub">Employee sentiment score</div>
        </div>''', unsafe_allow_html=True)
    with c4:
        att_color = C["red"] if att > 0.03 else C["orange"] if att > 0.02 else C["green_dark"]
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
            <div class="hero-label" style="color:{C['orange_dark']}">Monthly Attrition</div>
            <div class="hero-value" style="color:{att_color}">{att:.1%}</div>
            <div class="hero-sub">Avg monthly turnover rate</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # Architecture
    st.markdown('<h2><span class="gradient-title">System Architecture</span></h2>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns([1, 0.2, 1], gap="medium")
    with a1:
        st.markdown(f'''<div class="acard">
            <div class="acard-label" style="color:{C['blue']}">Agent 1</div>
            <div class="acard-title">Talent Intelligence</div>
            <div class="acard-desc">
                <strong style="color:{C['blue']}">Routed NER</strong> Ensemble + <strong style="color:{C['purple']}">SBERT</strong> Matching<br>
                DistilBERT v6 &middot; GLiNER &middot; ModernBERT v11 &middot; gazetteer &middot; 10 entity types
            </div>
        </div>''', unsafe_allow_html=True)
    with a2:
        st.markdown('''<div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:16px">
            <div style="color:#86868B;font-size:0.8rem;font-weight:600;text-align:center;line-height:2">
                &larr;<br>Proprietary<br>Brain<br>&rarr;
            </div>
        </div>''', unsafe_allow_html=True)
    with a3:
        st.markdown(f'''<div class="acard">
            <div class="acard-label" style="color:{C['orange']}">Agent 2</div>
            <div class="acard-title">Workforce Forecasting</div>
            <div class="acard-desc">
                <strong style="color:{C['orange']}">Bi-LSTM</strong> Attrition Model<br>
                Temporal pattern analysis &middot; Headcount projections &middot; Risk alerts
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # Two-column charts
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig = go.Figure()
        for dept in monthly["department"].unique():
            d = monthly[monthly["department"] == dept].sort_values("month")
            fig.add_trace(go.Scatter(x=d["month"], y=d["headcount"], mode="lines",
                name=dept, line=dict(color=DEPT_COLORS.get(dept,"#86868B"), width=2.5),
                hovertemplate=f"<b>{dept}</b><br>Month %{{x}}<br>HC: %{{y}}<extra></extra>"))
        fig.update_layout(title="Headcount by Department")
        style_fig(fig, 420)
        fig.update_layout(
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        font=dict(size=11)),
            margin=dict(l=20, r=120, t=48, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Workforce composition donut
        comp = cur.set_index("department")["headcount"]
        fig = go.Figure(go.Pie(
            labels=comp.index, values=comp.values, hole=0.65,
            marker=dict(colors=[DEPT_COLORS.get(d,"#86868B") for d in comp.index],
                        line=dict(color="#fff", width=2)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Headcount: %{value}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(title="Workforce Composition", showlegend=False,
                          annotations=[dict(text=f"<b>{hc}</b><br>Total", x=0.5, y=0.5,
                                            font_size=18, showarrow=False, font_color="#1D1D1F")])
        style_fig(fig, 380)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close dot-bg


# ─── SKILL-GAP ANALYZER TAB ─────────────────────────────────────────
def render_skill_gap_tab():
    """Third tab on the Talent Intelligence page — surfaces the full
    analyzer output from tools.talent_tools.analyze_skill_gap."""
    from data.sample_resumes import get_active_resumes, get_active_jds
    SAMPLE_RESUMES = get_active_resumes()
    SAMPLE_JOB_DESCRIPTIONS = get_active_jds()
    import json as _json

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        sel_c = st.selectbox("Candidate", [r["name"] for r in SAMPLE_RESUMES],
                             key="sg_c")
    with c2:
        sel_j = st.selectbox("Position", [j["title"] for j in SAMPLE_JOB_DESCRIPTIONS],
                             key="sg_j")

    cand = next(r for r in SAMPLE_RESUMES if r["name"] == sel_c)
    job = next(j for j in SAMPLE_JOB_DESCRIPTIONS if j["title"] == sel_j)

    run = st.button("Run Deep Analysis", key="sg_btn")
    if not run:
        st.markdown(
            '''<div class="section-card" style="min-height:220px;display:flex;
            flex-direction:column;align-items:center;justify-content:center;gap:14px">
                <div style="width:48px;height:48px;border-radius:14px;
                background:linear-gradient(135deg,#EBF5FF,#F5EEFA);display:flex;
                align-items:center;justify-content:center;font-size:1.5rem">⚙</div>
                <div style="text-align:center;color:#86868B;font-size:0.85rem;
                line-height:1.6">
                Click <b>Run Deep Analysis</b> to score candidate-job fit across<br>
                19 hiring-intelligence signals using the Routed NER + SBERT pipeline</div>
            </div>''',
            unsafe_allow_html=True,
        )
        return

    ner = load_ner()
    sbert = load_sbert()
    if ner is None or sbert is None:
        st.error("NER or SBERT model not available. "
                 "Run `python main.py --mode train` first.")
        return

    with st.spinner("Running ensemble NER + SBERT diff + Hungarian matching…"):
        from tools.talent_tools import analyze_skill_gap
        raw = analyze_skill_gap.run(candidate_text=cand["text"], job_text=job["text"])
        result = _json.loads(raw)

    if "error" in result:
        st.error(result["error"])
        return

    fit = result.get("fit_score", {})
    score = fit.get("composite_fit_score", 0)
    tier = fit.get("recommendation_tier", "?")
    components = fit.get("components", {})
    coverage = result.get("coverage", {})
    years = result.get("years_experience_gap", {}) or {}
    sen = result.get("seniority_alignment", {}) or {}
    pedigree = result.get("employer_pedigree", {}) or {}
    impact = result.get("resume_impact_signals", []) or []
    confidence = result.get("confidence", {}) or {}
    coaching = result.get("coaching_paths", []) or []
    gaps = result.get("gaps_by_type", {}) or {}
    briefing = result.get("executive_briefing", {}) or {}

    # --------- Row 1: hero metrics ---------
    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        fc = _fit_color(score)
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Composite Fit</div>
            <div class="hero-value" style="color:{fc}">{score:.0f}<span style="font-size:1.2rem;color:#86868B">/100</span></div>
            <div class="hero-sub">4-signal weighted score</div>
        </div>''', unsafe_allow_html=True)
    with h2:
        st.markdown(f'''<div class="hero-card" style="display:flex;flex-direction:column;
            justify-content:center;align-items:flex-start">
            <div class="hero-label" style="color:{C['purple']}">Recommendation</div>
            <div style="margin:14px 0 6px 0">{_tier_badge(tier)}</div>
            <div class="hero-sub">Hiring tier</div>
        </div>''', unsafe_allow_html=True)
    with h3:
        if years.get("meets_requirement") is True:
            val = f'+{years.get("delta_years", 0)}'
            yc = C["green_dark"]
            sub = f'{years.get("candidate_has")} vs {years.get("required")}'
        elif years.get("meets_requirement") is False:
            val = f'{years.get("delta_years", 0)}'
            yc = C["red"]
            sub = f'{years.get("candidate_has")} vs {years.get("required")}'
        else:
            val = "—"
            yc = C["text2"]
            sub = "No explicit requirement"
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_green']}"></div>
            <div class="hero-label" style="color:{C['green_dark']}">Experience Delta</div>
            <div class="hero-value" style="color:{yc}">{val}<span style="font-size:1.1rem;color:#86868B"> yrs</span></div>
            <div class="hero-sub">{sub}</div>
        </div>''', unsafe_allow_html=True)
    with h4:
        align = sen.get("alignment", "—")
        align_label = align.replace("_", " ").title()
        align_color = C["green_dark"] if align == "aligned" else C["orange"]
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Seniority</div>
            <div class="hero-value" style="color:{align_color};font-size:1.6rem">{align_label}</div>
            <div class="hero-sub">{sen.get("candidate_level_label", "?")} vs {sen.get("job_level_label", "?")}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # --------- Row 1.5: Policy sensitivity (cutoff variants + confidence band) ---------
    band_pp = _confidence_to_band_pp(confidence.get("score"))
    score_low = max(0, score - band_pp)
    score_high = min(100, score + band_pp)
    conf_lvl = (confidence.get("level") or "—").capitalize()
    band_color = (C["green_dark"] if conf_lvl.lower() == "high"
                   else C["orange"] if conf_lvl.lower() == "medium"
                   else C["red"])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Policy Sensitivity")
    st.markdown(
        '<div style="color:#86868B;font-size:0.82rem;margin-bottom:14px">'
        'How robust is this verdict to plausible cutoff and weighting '
        'choices? The bar shows the fit-score confidence interval; the '
        'three pills show what tier the candidate would land in under '
        'three different cutoff policies.</div>',
        unsafe_allow_html=True,
    )

    ps_l, ps_r = st.columns([1.1, 1.3], gap="large")
    with ps_l:
        # Confidence band on the fit score — visualized as a horizontal range bar.
        band_html = f'''
        <div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;
            letter-spacing:0.06em;font-weight:600;margin-bottom:6px">
            Fit Score with confidence band
        </div>
        <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:12px">
            <div style="font-size:2rem;font-weight:800;color:{_fit_color(score)};
                letter-spacing:-0.03em;line-height:1">{score:.0f}</div>
            <div style="font-size:0.85rem;color:#86868B">/100 ·
                <span style="color:{band_color};font-weight:600">±{band_pp:.1f}pp</span>
                ({conf_lvl} confidence)</div>
        </div>
        <div style="position:relative;height:14px;background:#F0F0F5;border-radius:7px;overflow:hidden">
            <div style="position:absolute;left:{score_low:.1f}%;width:{(score_high - score_low):.1f}%;
                top:0;bottom:0;background:linear-gradient(90deg,
                {_fit_color(score_low)} 0%, {_fit_color(score)} 50%,
                {_fit_color(score_high)} 100%);opacity:0.55;border-radius:7px"></div>
            <div style="position:absolute;left:calc({score:.1f}% - 1px);top:-4px;
                width:3px;height:22px;background:#1D1D1F;border-radius:1px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;
            color:#86868B;margin-top:6px">
            <span>{score_low:.1f} (low)</span>
            <span>{score:.1f} (point estimate)</span>
            <span>{score_high:.1f} (high)</span>
        </div>'''
        st.markdown(band_html, unsafe_allow_html=True)
    with ps_r:
        st.markdown(
            '<div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;'
            'letter-spacing:0.06em;font-weight:600;margin-bottom:8px">'
            'Tier under different policies</div>',
            unsafe_allow_html=True,
        )
        for key, policy in TIER_CUTOFF_POLICIES.items():
            t_lo = _score_to_tier(score_low, key)
            t_pt = _score_to_tier(score, key)
            t_hi = _score_to_tier(score_high, key)
            same_across_band = (t_lo == t_pt == t_hi)
            stability_icon = "✓" if same_across_band else "△"
            stability_color = C["green_dark"] if same_across_band else C["orange"]
            stability_note = ("stable across band"
                              if same_across_band
                              else f"shifts to {t_lo.replace('_',' ')}/{t_hi.replace('_',' ')} at band edges")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;'
                f'padding:8px 0;border-bottom:1px solid #F0F0F5">'
                f'<div style="min-width:64px;font-size:0.78rem;font-weight:600;'
                f'color:#1D1D1F">{policy["label"]}</div>'
                f'<div style="flex:1">{_tier_badge(t_pt)}</div>'
                f'<div style="font-size:0.72rem;color:{stability_color};'
                f'font-weight:600">{stability_icon} {stability_note}</div>'
                f'</div>'
                f'<div style="font-size:0.7rem;color:#86868B;padding-left:76px;'
                f'margin-bottom:6px">'
                f'≥{policy["thresholds"][0]} STRONG_HIRE · '
                f'≥{policy["thresholds"][1]} HIRE · '
                f'≥{policy["thresholds"][2]} INTERVIEW · '
                f'≥{policy["thresholds"][3]} CONDITIONAL'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # --------- Row 2: Executive briefing + coaching paths ---------
    b1, b2 = st.columns([1.3, 1], gap="large")
    with b1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Executive Briefing")
        verdict = briefing.get("verdict_paragraph", "")
        strengths = briefing.get("strengths_paragraph", "")
        concerns = briefing.get("concerns_paragraph", "")
        if verdict:
            st.markdown(f'<div class="briefing-block briefing-verdict">{verdict}</div>',
                        unsafe_allow_html=True)
        if strengths:
            # Convert inline markdown-ish hyphens to line breaks
            strengths_html = strengths.replace("  - ", "<br>&bull; ")
            st.markdown(f'<div class="briefing-block briefing-strengths">{strengths_html}</div>',
                        unsafe_allow_html=True)
        if concerns:
            concerns_html = concerns.replace("  - ", "<br>&bull; ")
            st.markdown(f'<div class="briefing-block briefing-concerns">{concerns_html}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Coaching Paths")
        if coaching:
            for p in coaching:
                delta = p.get("delta", 0)
                cur_fit = p.get("current_fit", 0)
                proj = p.get("projected_fit", 0)
                proj_tier = p.get("projected_tier", "?")
                crit = p.get("criticality", "STANDARD")
                st.markdown(f'''<div class="coach-row">
                    <div class="coach-row-title">Close: {p.get("if_candidate_closes_gap", "?")} {_crit_badge(crit)}</div>
                    <div class="coach-row-delta">Fit {cur_fit:.0f} → {proj:.0f} <span class="delta-up">(+{delta:.1f})</span> → {proj_tier.replace("_", " ")}</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#86868B;font-size:0.85rem;padding:20px 0">'
                        'No coaching paths identified — candidate already covers top requirements.</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # --------- Row 3: Fit-score components ---------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Fit Score Breakdown")
    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.markdown(
            _fit_bar("Critical-weighted coverage (60%)",
                     components.get("critical_weighted_coverage", 0), C["blue"])
            + _fit_bar("Overall coverage (15%)",
                       components.get("overall_coverage", 0), C["purple"]),
            unsafe_allow_html=True,
        )
    with cc2:
        st.markdown(
            _fit_bar("Years of experience (15%)",
                     components.get("years_score", 0), C["green_dark"])
            + _fit_bar("Seniority alignment (10%)",
                       components.get("seniority_score", 0), C["orange"]),
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # --------- Row 4: Gap breakdown per entity type ---------
    sub_tabs = st.tabs(["Skills", "Certifications", "Degrees"])
    for st_tab, etype, label in zip(sub_tabs, ["SKILL", "CERT", "DEGREE"],
                                     ["Skills", "Certifications", "Degrees"]):
        with st_tab:
            bucket = gaps.get(etype, {})
            matched = bucket.get("matched", [])
            weak = bucket.get("weak", [])
            missing = bucket.get("missing", [])
            cov_pct = coverage.get(f"{etype.lower()}_coverage_pct", 0)

            st.markdown(
                f'<div style="color:#86868B;font-size:0.82rem;margin-bottom:10px">'
                f'{label} coverage: <strong style="color:#1D1D1F">{cov_pct:.0f}%</strong> '
                f'&middot; {len(matched)} matched &middot; {len(weak)} weak &middot; '
                f'{len(missing)} missing</div>',
                unsafe_allow_html=True,
            )
            if not (matched or weak or missing):
                st.markdown(
                    '<div style="color:#86868B;font-size:0.85rem;padding:16px 0">'
                    f'No {label.lower()} requirements extracted from this JD.</div>',
                    unsafe_allow_html=True,
                )
                continue

            gc1, gc2, gc3 = st.columns(3, gap="medium")
            with gc1:
                st.markdown(f'**Matched ({len(matched)})**')
                for m in matched:
                    st.markdown(_render_gap_item(m), unsafe_allow_html=True)
                if not matched:
                    st.markdown('<span style="color:#86868B;font-size:0.82rem">—</span>',
                                unsafe_allow_html=True)
            with gc2:
                st.markdown(f'**Weak ({len(weak)})**')
                for m in weak:
                    st.markdown(_render_gap_item(m), unsafe_allow_html=True)
                if not weak:
                    st.markdown('<span style="color:#86868B;font-size:0.82rem">—</span>',
                                unsafe_allow_html=True)
            with gc3:
                st.markdown(f'**Missing ({len(missing)})**')
                for m in missing:
                    st.markdown(_render_gap_item(m, is_missing=True),
                                unsafe_allow_html=True)
                if not missing:
                    st.markdown('<span style="color:#86868B;font-size:0.82rem">—</span>',
                                unsafe_allow_html=True)

    st.markdown("")

    # --------- Row 5: Evidence, pedigree, confidence ---------
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        st.markdown('<div class="section-card" style="min-height:180px">',
                    unsafe_allow_html=True)
        st.markdown("### Tier-1 Pedigree")
        t1_count = pedigree.get("tier_1_count", 0)
        total_emp = pedigree.get("total_employers_extracted", 0)
        if t1_count > 0:
            st.markdown(
                f'<div style="font-size:2rem;font-weight:800;color:#248A3D;'
                f'letter-spacing:-0.03em;margin:4px 0 12px 0">'
                f'{t1_count}<span style="font-size:1rem;color:#86868B">/{total_emp}</span></div>',
                unsafe_allow_html=True,
            )
            for h in pedigree.get("tier_1_employers", [])[:5]:
                st.markdown(
                    f'<span class="etag etag-employer">{h.get("employer", "?")}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem">'
                'No tier-1 energy employers identified in candidate history.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="section-card" style="min-height:180px">',
                    unsafe_allow_html=True)
        st.markdown("### Quantified Impact")
        if impact:
            st.markdown(
                f'<div style="color:#86868B;font-size:0.8rem;margin-bottom:10px">'
                f'{len(impact)} quantified achievement'
                f'{"s" if len(impact) != 1 else ""} in resume</div>',
                unsafe_allow_html=True,
            )
            for q in impact[:8]:
                st.markdown(f'<span class="etag etag-years">{q}</span>',
                            unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem">'
                'No quantified figures extracted from the resume.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with e3:
        st.markdown('<div class="section-card" style="min-height:180px">',
                    unsafe_allow_html=True)
        st.markdown("### Analysis Confidence")
        lvl = confidence.get("level", "—")
        sco = confidence.get("score", 0)
        lvl_color = (C["green_dark"] if lvl == "high"
                     else C["orange"] if lvl == "medium"
                     else C["red"])
        st.markdown(
            f'<div style="font-size:2rem;font-weight:800;color:{lvl_color};'
            f'letter-spacing:-0.03em;margin:4px 0 8px 0;text-transform:capitalize">'
            f'{lvl}</div>'
            f'<div style="color:#86868B;font-size:0.78rem;margin-bottom:10px">'
            f'Score: {sco:.2f} · Taxonomy hits: {result.get("taxonomy_hits", 0)}</div>',
            unsafe_allow_html=True,
        )
        for r in confidence.get("reasons", [])[:3]:
            st.markdown(
                f'<div style="color:#6E6E73;font-size:0.78rem;line-height:1.5;'
                f'margin-bottom:6px">· {r}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ─── TALENT PAGE ────────────────────────────────────────────────────
@st.cache_data
def _load_unified_eval():
    """Load the unified-evaluator output from the canonical-sidecar run.

    Returns ``None`` when the file isn't present (i.e., the eval hasn't
    been run yet). The dashboard hides the Model Quality panel in that
    case rather than fabricate metrics. To regenerate run:
        ./venv/bin/python scripts/eval_unified.py
    """
    path = DATA_DIR / "unified_eval_latest.json"
    if not path.exists():
        return None
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        try:
            import json as _json
            return _json.loads(path.read_text())
        except Exception:
            return None


def _render_model_quality_panel():
    """Render the Model Quality panel on the Talent Intelligence page.

    Shows per-class F1 from the unified evaluator (lenient + strict),
    per-model comparison, and an honest caveat block. Source of truth is
    `data/processed/unified_eval_latest.json` which is produced by
    `scripts/eval_unified.py` against the canonical adjudicated sidecar.
    """
    eval_data = _load_unified_eval()
    if not eval_data:
        return  # eval not run; silently hide the panel

    metrics = eval_data.get("metrics", {})
    label_schema = eval_data.get("label_schema", [])
    n_docs = eval_data.get("n_docs", 0)
    n_gold = eval_data.get("n_gold_spans_total", 0)
    ts = eval_data.get("generated_at_utc", "—")

    routed_strict = (metrics.get("routed_production", {})
                     .get("strict", {}).get("__overall_micro__", {}))
    routed_lenient = (metrics.get("routed_production", {})
                      .get("lenient", {}).get("__overall_micro__", {}))

    with st.expander(
        f"Model Quality — Routed NER strict-F1 "
        f"{routed_strict.get('f1', 0):.2f} · "
        f"lenient-F1 {routed_lenient.get('f1', 0):.2f} "
        f"(eval on canonical adjudicated dev set, "
        f"{n_docs} docs / {n_gold} gold spans)",
        expanded=False,
    ):
        # ---- Headline KPIs (routed_production) ----
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            f1_l = routed_lenient.get("f1", 0)
            color = (C["green_dark"] if f1_l >= 0.6
                     else C["orange"] if f1_l >= 0.3
                     else C["red"])
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
                <div class="hero-label" style="color:{C['blue']}">Lenient F1</div>
                <div class="hero-value" style="color:{color}">{f1_l:.2f}</div>
                <div class="hero-sub">Overlap match · routed production</div>
            </div>''', unsafe_allow_html=True)
        with m2:
            f1_s = routed_strict.get("f1", 0)
            color = (C["green_dark"] if f1_s >= 0.5
                     else C["orange"] if f1_s >= 0.2
                     else C["red"])
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
                <div class="hero-label" style="color:{C['purple']}">Strict F1</div>
                <div class="hero-value" style="color:{color}">{f1_s:.2f}</div>
                <div class="hero-sub">Exact-match · routed production</div>
            </div>''', unsafe_allow_html=True)
        with m3:
            p = routed_lenient.get("precision", 0)
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_green']}"></div>
                <div class="hero-label" style="color:{C['green_dark']}">Precision</div>
                <div class="hero-value">{p:.2f}</div>
                <div class="hero-sub">Lenient · % predictions correct</div>
            </div>''', unsafe_allow_html=True)
        with m4:
            r = routed_lenient.get("recall", 0)
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
                <div class="hero-label" style="color:{C['orange_dark']}">Recall</div>
                <div class="hero-value">{r:.2f}</div>
                <div class="hero-sub">Lenient · % gold spans found</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("")

        # ---- Per-class table for routed_production ----
        st.markdown("### Per-class quality — Routed NER (production path)")
        st.markdown(
            '<div style="color:#86868B;font-size:0.82rem;margin-bottom:8px">'
            'Lenient = any character-overlap counts as a match. Strict = '
            'exact-match on label + offsets. Both reported here so the '
            'gap shows where the model is right but boundary-noisy.</div>',
            unsafe_allow_html=True,
        )
        rows_html = []
        for L in label_schema:
            ml = (metrics.get("routed_production", {})
                  .get("lenient", {}).get(L, {}))
            ms = (metrics.get("routed_production", {})
                  .get("strict", {}).get(L, {}))
            f1_l = ml.get("f1", 0)
            f1_s = ms.get("f1", 0)
            p = ml.get("precision", 0)
            r = ml.get("recall", 0)
            gold_support = ml.get("support_gold", 0)
            color = (C["green_dark"] if f1_l >= 0.5
                     else C["orange"] if f1_l >= 0.2
                     else C["red"] if gold_support >= 5
                     else C["text2"])
            rows_html.append(
                f'<div class="drow" style="border-left:3px solid {color}">'
                f'<div><div class="drow-name">{L}</div>'
                f'<div class="drow-meta">{gold_support} gold spans</div></div>'
                f'<div style="display:flex;gap:18px;align-items:center">'
                f'<div style="text-align:center"><div style="font-size:0.7rem;color:#86868B">P</div>'
                f'<div style="font-weight:700;color:#1D1D1F;font-size:0.85rem">{p:.2f}</div></div>'
                f'<div style="text-align:center"><div style="font-size:0.7rem;color:#86868B">R</div>'
                f'<div style="font-weight:700;color:#1D1D1F;font-size:0.85rem">{r:.2f}</div></div>'
                f'<div style="text-align:center"><div style="font-size:0.7rem;color:#86868B">Lenient F1</div>'
                f'<div style="font-weight:800;color:{color};font-size:1.0rem">{f1_l:.2f}</div></div>'
                f'<div style="text-align:center"><div style="font-size:0.7rem;color:#86868B">Strict F1</div>'
                f'<div style="font-weight:600;color:#86868B;font-size:0.85rem">{f1_s:.2f}</div></div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(
            '<div class="section-card">' + "".join(rows_html) + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ---- Per-model comparison chart (lenient overall F1) ----
        models_in_order = ["routed_production", "v6_ensemble", "v11_modernbert",
                           "gliner_10type_candidate", "gazetteer_only"]
        rows = []
        for m in models_in_order:
            mm = metrics.get(m, {})
            ovr = mm.get("lenient", {}).get("__overall_micro__", {}) or {}
            rows.append({
                "model": m,
                "lenient_f1": ovr.get("f1", 0),
                "lenient_p": ovr.get("precision", 0),
                "lenient_r": ovr.get("recall", 0),
                "avg_inference_ms": mm.get("avg_inference_ms", 0),
            })
        df = pd.DataFrame(rows)
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            colors = [C["green_dark"] if m == "routed_production" else C["blue"]
                      for m in df["model"][::-1]]
            fig = go.Figure(go.Bar(
                y=list(df["model"][::-1]),
                x=list(df["lenient_f1"][::-1]),
                orientation="h",
                marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
                text=[f"{v:.3f}" for v in df["lenient_f1"][::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Lenient F1: %{x:.3f}<extra></extra>",
            ))
            fig.update_layout(
                title="Lenient overall F1 by model",
                xaxis=dict(range=[0, max(0.2, df["lenient_f1"].max() * 1.4)],
                            title="F1"),
                yaxis=dict(title=""),
            )
            style_fig(fig, max(280, 60 + 38 * len(df)))
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Per-model trade-off")
            for _, row in df.iterrows():
                m = row["model"]
                bcol = C["green_dark"] if m == "routed_production" else C["text2"]
                st.markdown(
                    f'<div style="padding:6px 0;border-bottom:1px solid #F0F0F5">'
                    f'<div style="font-size:0.85rem;font-weight:600;color:{bcol}">{m}</div>'
                    f'<div style="font-size:0.75rem;color:#86868B">'
                    f'P={row["lenient_p"]:.3f} · R={row["lenient_r"]:.3f} '
                    f'· ~{row["avg_inference_ms"]:.0f}ms/doc'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")

        # ---- Honest caveat block ----
        st.markdown(
            f'<div class="section-card" style="border-left:3px solid {C["orange"]}">'
            f'<div style="font-weight:700;color:#1D1D1F;font-size:0.9rem;margin-bottom:8px">'
            f'How to read these numbers</div>'
            f'<div style="font-size:0.82rem;color:#1D1D1F;line-height:1.6">'
            f'Evaluated on the <strong>canonical adjudicated dev set</strong> '
            f'({n_docs} real resumes, {n_gold} human-reviewed gold spans, '
            f'10-class v7 schema). The current absolute F1 is low because:'
            f'<ul style="margin:8px 0 0 16px;padding:0;color:#6E6E73">'
            f'<li><strong>Truncation</strong>: v11 ModernBERT truncates at 512 tokens '
            f'and GLiNER at 384, but real resumes in this dev set average ~6500 '
            f'characters — most of each document is unseen by the models. '
            f'Fixing this is Step 3 of the documented training plan.</li>'
            f'<li><strong>Domain shift</strong>: v6 DistilBERT was trained on synthetic '
            f'energy-sector resumes; the dev set is real cross-industry resumes '
            f'from the snehaanbhawal corpus.</li>'
            f'<li><strong>Boundary noise</strong>: gold annotations use tight, '
            f'human-conservative spans. Models often predict broader phrases that '
            f'overlap but don\'t exact-match — visible as the gap between '
            f'lenient and strict F1.</li>'
            f'</ul>'
            f'<div style="margin-top:10px;color:#86868B;font-size:0.74rem">'
            f'Eval generated {ts} from <code>scripts/eval_unified.py</code>. '
            f'Re-run with <code>./venv/bin/python scripts/eval_unified.py</code> '
            f'after model changes. Discipline: <code>eval_blind_v1.json</code> was '
            f'NOT accessed by this evaluator.</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_talent_page(employees, monthly):
    st.markdown('<h1><span class="gradient-title">Talent Intelligence</span></h1>', unsafe_allow_html=True)
    st.markdown("AI-powered resume analysis and candidate-job matching using a Routed NER ensemble (DistilBERT v6 + GLiNER + ModernBERT v11 + gazetteer, 10 entity types) and SBERT embeddings. Quality benchmarked against the canonical adjudicated dev set (840 gold spans across 25 real resumes).")
    st.markdown("")

    _render_model_quality_panel()

    tab1, tab2, tab3 = st.tabs(["Resume Analysis", "Candidate Ranking", "Skill Gap Analysis"])

    with tab1:
        from data.sample_resumes import get_active_resumes
        SAMPLE_RESUMES = get_active_resumes()
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            st.markdown("### Select Resume")
            sel = st.selectbox("Candidate", [r["name"] for r in SAMPLE_RESUMES], label_visibility="collapsed")
            resume = next(r for r in SAMPLE_RESUMES if r["name"] == sel)
            st.text_area("Resume text", resume["text"], height=260, disabled=True, label_visibility="collapsed")
            st.markdown("")
            run = st.button("Extract Entities")

        with c2:
            cache_key = f"ner_result__{sel}"
            if run:
                with st.spinner("Loading Routed NER ensemble (DistilBERT v6 + GLiNER + ModernBERT v11 + gazetteer, ~15-30s first time)..."):
                    ner = load_ner()
                if ner is None:
                    st.error("NER model not available. Run `python main.py --mode train` first.")
                    st.session_state.pop(cache_key, None)
                else:
                    with st.spinner("Running Routed NER (DistilBERT v6 + GLiNER + ModernBERT v11 + gazetteer)..."):
                        st.session_state[cache_key] = ner.extract_entities(resume["text"])

            ents = st.session_state.get(cache_key)
            if ents:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown("### Extracted Entities")
                total = sum(len(v) for v in ents.values())
                st.markdown(f'<div style="color:#86868B;font-size:0.82rem;margin-bottom:12px">{total} entities found</div>',
                            unsafe_allow_html=True)
                # Renders whatever types the loaded model produced — v6 returns 5,
                # v7 returns 10 (TOOL/INDUSTRY/LOCATION/PROJECT/SOFT_SKILL).
                display_order = [
                    ("Skills", "SKILL", "etag-skill"),
                    ("Tools", "TOOL", "etag-skill"),
                    ("Certifications", "CERT", "etag-cert"),
                    ("Education", "DEGREE", "etag-degree"),
                    ("Employers", "EMPLOYER", "etag-employer"),
                    ("Experience", "YEARS_EXP", "etag-years"),
                    ("Industry", "INDUSTRY", "etag-employer"),
                    ("Location", "LOCATION", "etag-employer"),
                    ("Projects", "PROJECT", "etag-employer"),
                    ("Soft Skills", "SOFT_SKILL", "etag-skill"),
                ]
                for label, key, cls in display_order:
                    items = ents.get(key, [])
                    if items:
                        st.markdown(f"**{label}**")
                        st.markdown("".join(f'<span class="etag {cls}">{i}</span>' for i in items), unsafe_allow_html=True)
                        st.markdown("")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('''<div class="section-card" style="min-height:260px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px">
                    <div style="width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#EBF5FF,#F5EEFA);display:flex;align-items:center;justify-content:center;font-size:1.5rem">&#x1F50D;</div>
                    <div style="text-align:center;color:#86868B;font-size:0.85rem;line-height:1.6">
                    Click <b>Extract Entities</b> to analyze the resume<br>using the Routed NER ensemble (DistilBERT v6 + GLiNER + ModernBERT v11 + gazetteer)</div>
                    <div style="width:80%;max-width:280px">
                        <div class="shimmer-line" style="width:100%;height:10px;margin-bottom:6px"></div>
                        <div class="shimmer-line" style="width:70%;height:10px;margin-bottom:6px"></div>
                        <div class="shimmer-line" style="width:85%;height:10px"></div>
                    </div>
                </div>''', unsafe_allow_html=True)

    with tab2:
        from data.sample_resumes import get_active_resumes, get_active_jds
        SAMPLE_RESUMES = get_active_resumes()
        SAMPLE_JOB_DESCRIPTIONS = get_active_jds()
        st.markdown("### Select Position")
        sel_job = st.selectbox("Job", [j["title"] for j in SAMPLE_JOB_DESCRIPTIONS], label_visibility="collapsed")
        job = next(j for j in SAMPLE_JOB_DESCRIPTIONS if j["title"] == sel_job)
        st.text_area("Job description", job["text"], height=120, disabled=True, label_visibility="collapsed", key="jd")

        sbert = load_sbert()
        if sbert:
            with st.spinner("Computing SBERT embeddings..."):
                rankings = sbert.rank_candidates(SAMPLE_RESUMES, job)
            st.markdown("")

            g1, g2 = st.columns([1.3, 1], gap="large")
            with g1:
                st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
                st.markdown(f"### Candidate Rankings")
                for i, r in enumerate(rankings, 1):
                    sc = score_color(r["match_score"])
                    pct = r["match_score"] * 100
                    st.markdown(f'''<div class="mrow">
                        <div class="mrow-rank">#{i}</div>
                        <div><div class="mrow-name">{r["candidate_name"]}</div>
                        <div class="mrow-tier">{r["match_tier"]}</div></div>
                        <div class="mrow-score" style="color:{sc}">{pct:.0f}%</div>
                    </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with g2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                df = pd.DataFrame(rankings)
                fig = go.Figure(go.Bar(
                    y=df["candidate_name"][::-1], x=df["match_score"][::-1],
                    orientation="h",
                    marker=dict(color=[score_color(s) for s in df["match_score"][::-1]],
                                cornerradius=6, line=dict(width=0)),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.1%}<extra></extra>",
                ))
                fig.add_vline(x=0.60, line_dash="dot", line_color="#D2D2D7",
                              annotation=dict(text="Threshold", font=dict(color="#86868B", size=10)))
                fig.update_layout(title="Match Score Distribution", xaxis=dict(range=[0,1]),
                                  xaxis_title="Cosine Similarity", yaxis_title="")
                style_fig(fig, 380)
                st.plotly_chart(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        render_skill_gap_tab()


# ─── FORECAST PAGE ──────────────────────────────────────────────────
def render_forecast_page(monthly, individual):
    st.markdown('<h1><span class="gradient-title">Workforce Forecasting</span></h1>', unsafe_allow_html=True)
    st.markdown("Bi-LSTM neural network analyzes 12-month temporal patterns to predict attrition risk and project headcount.")
    st.markdown("")

    forecast = load_forecast()
    tab1, tab2, tab3, tab4 = st.tabs([
        "Risk Overview", "Department Drill-Down", "Headcount Projections",
        "Deep Risk Analysis",
    ])

    with tab1:
        if forecast:
            from tools.forecast_tools import set_forecast_engine, set_workforce_data
            set_forecast_engine(forecast)
            set_workforce_data(monthly, individual)
            preds = forecast.predict_all_departments(monthly)

            if preds:
                # Risk gauge + list
                g1, g2 = st.columns([1, 1.5], gap="large")
                with g1:
                    # Gauge for highest risk
                    top = preds[0]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=top["attrition_probability"] * 100,
                        number=dict(suffix="%", font=dict(size=48, color=C["text"])),
                        title=dict(text=f"Highest Risk: {top['department']}", font=dict(size=14, color=C["text2"])),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#D2D2D7"),
                            bar=dict(color=C["red"] if top["attrition_probability"] > 0.5 else C["orange"]),
                            bgcolor="#F0F0F0",
                            steps=[
                                dict(range=[0, 30], color="#E8FAF0"),
                                dict(range=[30, 60], color="#FFF8EC"),
                                dict(range=[60, 100], color="#FFF0EF"),
                            ],
                            threshold=dict(line=dict(color=C["red"], width=3), thickness=0.8, value=70),
                        ),
                    ))
                    style_fig(fig, 300)
                    st.plotly_chart(fig, width="stretch")

                with g2:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown("### All Departments")
                    for p in preds:
                        prob = p["attrition_probability"] * 100
                        bc = C["red"] if p["risk_level"] == "CRITICAL" else C["orange"] if p["risk_level"] == "HIGH" else C["green_dark"]
                        glow_cls = "drow-critical" if p["risk_level"] == "CRITICAL" else "drow-high" if p["risk_level"] == "HIGH" else ""
                        ring = _svg_ring(prob / 100, bc, 36)
                        st.markdown(f'''<div class="drow {glow_cls}">
                            <div><div class="drow-name">{p["department"]}</div>
                            <div class="drow-meta">HC: {p["current_headcount"]} &rarr; {p["projected_headcount_6m"]} (6-mo)</div></div>
                            <div style="display:flex;align-items:center;gap:14px">
                                <div class="ring-container">{ring}</div>
                                <div style="font-weight:700;font-size:0.9rem;color:{bc};min-width:42px;text-align:right">{prob:.0f}%</div>
                                {rbadge(p["risk_level"])}
                            </div>
                        </div>''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### Select Department")
        dept = st.selectbox("Dept", monthly["department"].unique(), label_visibility="collapsed", key="dd")
        dd = monthly[monthly["department"] == dept].sort_values("month")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.08,
                            subplot_titles=("Headcount","Monthly Departures","Satisfaction","Comp Ratio"))
        fig.add_trace(go.Scatter(x=dd["month"],y=dd["headcount"],mode="lines",
            line=dict(color=C["blue"],width=2.5),fill="tozeroy",fillcolor="rgba(0,113,227,0.06)"),row=1,col=1)
        fig.add_trace(go.Bar(x=dd["month"],y=dd["departures"],
            marker=dict(color=C["red"],cornerradius=3,opacity=0.8)),row=1,col=2)
        fig.add_trace(go.Scatter(x=dd["month"],y=dd["avg_satisfaction"],mode="lines",
            line=dict(color=C["green_dark"],width=2.5),fill="tozeroy",fillcolor="rgba(52,199,89,0.06)"),row=2,col=1)
        fig.add_trace(go.Scatter(x=dd["month"],y=dd["avg_comp_ratio"],mode="lines",
            line=dict(color=C["purple"],width=2.5),fill="tozeroy",fillcolor="rgba(175,82,222,0.06)"),row=2,col=2)
        fig.update_layout(showlegend=False, title=f"{dept} — 36-Month Trends")
        for a in fig.layout.annotations: a.font = dict(size=12, color=C["text2"])
        style_fig(fig, 520)
        fig.update_xaxes(gridcolor="#F0F0F0"); fig.update_yaxes(gridcolor="#F0F0F0")
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if forecast:
            preds = forecast.predict_all_departments(monthly)
            if preds:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                pdf = pd.DataFrame(preds)
                fig = go.Figure()
                for name, col, color in [("Current","current_headcount",C["blue"]),
                                          ("3-Month","projected_headcount_3m",C["teal"]),
                                          ("6-Month","projected_headcount_6m",C["orange"]),
                                          ("12-Month","projected_headcount_12m",C["red"])]:
                    fig.add_trace(go.Bar(name=name, x=pdf["department"], y=pdf[col],
                        marker=dict(color=color, cornerradius=4, opacity=0.9),
                        hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y}}<extra></extra>"))
                fig.update_layout(barmode="group", title="Headcount Projections by Department",
                                  xaxis_title="", yaxis_title="Headcount")
                style_fig(fig, 440)
                st.plotly_chart(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        render_deep_risk_tab(forecast, monthly, individual)


# ─── DEEP RISK ANALYSIS TAB — Agent 2's 19-signal workforce analyzer ─
def _risk_color(level):
    return {"CRITICAL": C["red"], "HIGH": C["orange"],
            "MEDIUM": C["purple"], "LOW": C["green_dark"]}.get(level, C["text2"])


def _sev_color(sev):
    return {"CRITICAL": C["red"], "HIGH": C["orange"],
            "MEDIUM": C["purple"], "LOW": C["green_dark"]}.get(sev, C["text2"])


def render_deep_risk_tab(forecast, monthly, individual):
    """Agent 2's 19-signal deep-risk analysis — the Bi-LSTM's probability
    decomposed into drivers, cohorts, simulated interventions, dollar
    exposure, and a boardroom-ready executive briefing. Mirrors Agent 1's
    Skill-Gap Analyzer tab architecture."""
    import json as _json

    if forecast is None:
        st.error("Bi-LSTM forecasting model not available. "
                 "Run `python main.py --mode train` first.")
        return

    # Ensure the forecast tools have access to the current data frames
    from tools.forecast_tools import set_forecast_engine, set_workforce_data
    set_forecast_engine(forecast)
    set_workforce_data(monthly, individual)

    st.markdown("")
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        dept = st.selectbox(
            "Department",
            sorted(monthly["department"].unique()),
            key="deep_risk_dept",
        )
    with c2:
        st.markdown(
            '<div style="padding-top:26px;color:#86868B;font-size:0.78rem;'
            'line-height:1.5">19-signal analysis using gradient attribution, '
            'cohort segmentation, and counterfactual Bi-LSTM forward passes.</div>',
            unsafe_allow_html=True,
        )

    run = st.button("Run Deep Workforce Analysis", key="deep_risk_btn")
    if not run:
        st.markdown(
            f'''<div class="section-card" style="min-height:220px;display:flex;
            flex-direction:column;align-items:center;justify-content:center;gap:14px">
                <div style="width:48px;height:48px;border-radius:14px;
                background:linear-gradient(135deg,#FFF0EF,#EBF5FF);display:flex;
                align-items:center;justify-content:center;font-size:1.5rem">△</div>
                <div style="text-align:center;color:#86868B;font-size:0.85rem;
                line-height:1.6">
                Click <b>Run Deep Workforce Analysis</b> to decompose {dept}'s<br>
                attrition risk across 19 hiring-intelligence signals — gradient<br>
                driver attribution, cohort segmentation, intervention simulator,<br>
                replacement cost, internal mobility bench, executive briefing</div>
            </div>''',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Running Bi-LSTM + gradient attribution + cohort segmentation + counterfactual forward pass…"):
        from tools.forecast_tools import analyze_workforce_risk
        raw = analyze_workforce_risk.run(department=dept)
        result = _json.loads(raw)

    if "error" in result:
        st.error(result["error"])
        return

    # Extract blocks
    prob = result.get("attrition_probability", 0.0)
    risk_level = result.get("risk_level", "MEDIUM")
    cur_hc = result.get("current_headcount", 0)
    pred_dep = result.get("predicted_departures_12mo", 0)
    proj_6m = result.get("projected_headcount_6m", cur_hc)
    drivers = result.get("drivers", {}) or {}
    cohorts = result.get("cohorts", {}) or {}
    trend = result.get("trend", {}) or {}
    seas = result.get("seasonality", {}) or {}
    leads = result.get("lead_indicators", {}) or {}
    cost = result.get("replacement_cost", {}) or {}
    vacancy = result.get("vacancy_exposure", {}) or {}
    knowledge = result.get("knowledge_loss", {}) or {}
    bench = result.get("internal_mobility", {}) or {}
    market = result.get("market_competition", {}) or {}
    plan = result.get("retention_plan", []) or []
    confidence = result.get("confidence", {}) or {}
    briefing = result.get("executive_briefing", {}) or {}

    # ---- Row 1: hero metrics ----
    h1, h2, h3, h4 = st.columns(4, gap="medium")
    rc = _risk_color(risk_level)
    with h1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_red']}"></div>
            <div class="hero-label" style="color:{rc}">Attrition Probability</div>
            <div class="hero-value" style="color:{rc}">{prob*100:.0f}<span style="font-size:1.2rem;color:#86868B">%</span></div>
            <div class="hero-sub">{risk_level} risk — {pred_dep} predicted departures over 12mo</div>
        </div>''', unsafe_allow_html=True)
    with h2:
        exposure = cost.get("estimated_replacement_cost_usd", 0)
        disp = f'${exposure/1000:.0f}K' if exposure < 1_000_000 else f'${exposure/1_000_000:.1f}M'
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
            <div class="hero-label" style="color:{C['orange']}">Replacement Exposure</div>
            <div class="hero-value" style="color:{C['orange_dark']}">{disp}</div>
            <div class="hero-sub">{vacancy.get('total_vacancy_days_at_risk', 0)} vacancy-days at risk</div>
        </div>''', unsafe_allow_html=True)
    with h3:
        top_drv = (drivers.get("top_drivers") or [{}])[0]
        drv_label = top_drv.get("label", "—")
        drv_pct = top_drv.get("pct_of_risk", 0)
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Primary Driver</div>
            <div class="hero-value" style="color:{C['purple']};font-size:1.5rem">{drv_label}</div>
            <div class="hero-sub">{drv_pct:.0f}% of model's attrition score</div>
        </div>''', unsafe_allow_html=True)
    with h4:
        sev = knowledge.get("severity", "LOW")
        sev_col = _sev_color(sev)
        tyr = knowledge.get("tenure_years_at_risk", 0)
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Knowledge-Loss</div>
            <div class="hero-value" style="color:{sev_col};font-size:1.5rem">{sev}</div>
            <div class="hero-sub">{tyr:.0f} tenure-years exposed to retirement</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ---- Row 2: Executive briefing + retention plan ----
    b1, b2 = st.columns([1.3, 1], gap="large")
    with b1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Executive Briefing")
        if briefing.get("verdict_paragraph"):
            st.markdown(
                f'<div class="briefing-block briefing-verdict">{briefing["verdict_paragraph"]}</div>',
                unsafe_allow_html=True,
            )
        if briefing.get("drivers_paragraph"):
            st.markdown(
                f'<div class="briefing-block briefing-strengths">{briefing["drivers_paragraph"]}</div>',
                unsafe_allow_html=True,
            )
        if briefing.get("action_paragraph"):
            st.markdown(
                f'<div class="briefing-block briefing-concerns">{briefing["action_paragraph"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Retention Plan")
        if plan:
            for p in plan:
                iv_label = p.get("intervention", "?").replace("_", " ").title()
                drv_label = p.get("target_driver_label", p.get("target_driver", "?"))
                cost_est = p.get("est_program_cost_usd", 0)
                lead = p.get("lead_time_days", 0)
                cohort_sz = p.get("cohort_size", 0)
                st.markdown(f'''<div class="coach-row">
                    <div class="coach-row-title">{iv_label} → {drv_label}</div>
                    <div class="coach-row-delta">{cohort_sz} people · ${cost_est:,} · {lead}d lead time</div>
                </div>''', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem;padding:20px 0">'
                'No targeted interventions — continue monitoring.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ---- Row 3: Driver attribution chart + Cohorts ----
    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Risk Driver Attribution")
        top_drvs = drivers.get("top_drivers", []) or []
        if top_drvs:
            labels = [d["label"] for d in top_drvs[:6]]
            values = [d["pct_of_risk"] for d in top_drvs[:6]]
            colors = [C["red"], C["orange"], C["purple"], C["blue"],
                      C["teal"], C["green_dark"]][:len(values)]
            fig = go.Figure(go.Bar(
                x=values, y=labels, orientation="h",
                marker=dict(color=colors, cornerradius=4),
                text=[f"{v:.0f}%" for v in values], textposition="outside",
                hovertemplate="<b>%{y}</b><br>Share of risk: %{x:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                title="Share of Bi-LSTM attrition score (gradient × input)",
                xaxis_title="% of risk", yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            style_fig(fig, 320)
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem">'
                'Driver attribution unavailable — insufficient history.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### At-Risk Cohorts")
        cohort_items = [
            ("retirement_cliff_critical", "Retirement Cliff (age ≥58)", C["red"]),
            ("retirement_cliff_emerging", "Retirement Emerging (55-57)", C["orange"]),
            ("new_hire_churn_window",     "New-Hire Churn (tenure <2y)", C["purple"]),
            ("comp_gap",                  "Comp Gap (ratio <0.90)", C["indigo"]),
            ("low_engagement",            "Low Engagement/Satisfaction", C["teal"]),
            ("tenured_ip",                "Tenured IP (≥8y)", C["green_dark"]),
        ]
        for key, label, color in cohort_items:
            c = cohorts.get(key, {})
            n = c.get("count", 0)
            pct = c.get("pct_of_department", 0)
            if n == 0:
                continue
            st.markdown(f'''<div class="drow" style="border-left:3px solid {color}">
                <div>
                    <div class="drow-name">{label}</div>
                    <div class="drow-meta">{n} people · {pct:.0f}% of department</div>
                </div>
                <div style="display:flex;align-items:center;gap:12px">
                    {_svg_ring(pct/100, color, 32)}
                    <div style="font-weight:700;color:{color};min-width:40px;text-align:right">{n}</div>
                </div>
            </div>''', unsafe_allow_html=True)
        hp_at_risk = cohorts.get("high_performer_at_risk", 0)
        if hp_at_risk:
            st.markdown(
                f'<div style="margin-top:12px;padding:10px 14px;background:#FFF0EF;'
                f'border-radius:10px;font-size:0.85rem;color:#D70015;font-weight:600">'
                f'⚠ {hp_at_risk} high-performer(s) are in a risk cohort — protect these first.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ---- Row 4: Intervention simulator + Internal mobility bench ----
    g1, g2 = st.columns([1.1, 1], gap="large")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Intervention Simulator")
        st.markdown(
            '<div style="color:#86868B;font-size:0.8rem;margin-bottom:10px">'
            'Counterfactual Bi-LSTM forward pass — "what would this lever do?"</div>',
            unsafe_allow_html=True,
        )
        sim_c1, sim_c2 = st.columns([2, 1], gap="small")
        with sim_c1:
            iv = st.selectbox(
                "Intervention",
                ["comp_adjustment", "satisfaction_program",
                 "engagement_initiative", "retention_bonus",
                 "flex_work", "knowledge_capture"],
                key="deep_risk_iv",
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with sim_c2:
            mag = st.number_input(
                "Magnitude", min_value=0.05, max_value=0.50,
                value=0.10, step=0.05, key="deep_risk_mag",
            )
        if st.button("Simulate", key="deep_risk_sim_btn"):
            from tools.forecast_tools import simulate_intervention
            sim_raw = simulate_intervention.run(
                department=dept, intervention=iv, magnitude=float(mag),
            )
            sim = _json.loads(sim_raw)
            if "error" in sim:
                st.error(sim["error"])
            else:
                p0 = sim.get("baseline_prob", 0) or 0
                p1 = sim.get("counterfactual_prob", 0) or 0
                delta = sim.get("delta_prob", 0) or 0
                est_cost = sim.get("estimated_program_cost_usd", 0)
                color = C["green_dark"] if delta < 0 else C["red"]
                arrow = "↓" if delta < 0 else "↑" if delta > 0 else "→"
                st.markdown(f'''<div style="padding:14px 18px;background:#FAFAFA;
                    border-radius:12px;border-left:3px solid {color}">
                    <div style="font-size:0.92rem;color:#1D1D1F;font-weight:600;margin-bottom:6px">
                        Baseline {p0:.1%} {arrow} Counterfactual {p1:.1%}
                        <span style="color:{color};font-weight:700">
                        ({delta*100:+.1f}pp)</span>
                    </div>
                    <div style="font-size:0.82rem;color:#6E6E73;line-height:1.5">
                        Modelled program cost: <b>${est_cost:,}</b> ·
                        Feature moved: <b>{sim.get('feature_moved', 'n/a')}</b>
                        ({sim.get('magnitude_raw', 0):+.2f} raw units) ·
                        {sim.get('lead_time_days', 0)}d lead time
                    </div>
                </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Internal Mobility Bench")
        top_c = bench.get("top_candidates", []) or []
        if top_c:
            for cand in top_c[:5]:
                mscore = cand.get("mobility_score", 0)
                bar_w = int(mscore * 100)
                st.markdown(f'''<div class="mrow">
                    <div class="mrow-rank">#{cand.get('employee_id', '?')}</div>
                    <div>
                        <div class="mrow-name">From {cand.get('from_department', '?')}</div>
                        <div class="mrow-tier">Tenure {cand.get('tenure_years', 0):.0f}y · Perf {cand.get('performance', 0):.1f} · Sat {cand.get('satisfaction', 0):.1f}</div>
                    </div>
                    <div class="mrow-score" style="color:{C['purple']}">{mscore:.2f}</div>
                </div>''', unsafe_allow_html=True)
            st.markdown(
                f'<div style="color:#86868B;font-size:0.78rem;margin-top:8px">'
                f'Ranking: adjacency × portability (tenure / perf / satisfaction / engagement).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem;padding:20px 0">'
                'No adjacent-department candidates — consider external sourcing.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ---- Row 5: Temporal & market signals ----
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        dir_ = trend.get("trend_direction", "—")
        rc2 = (C["red"] if dir_ == "accelerating"
               else C["green_dark"] if dir_ == "decelerating"
               else C["text2"])
        rn = trend.get("recent_6mo_attrition_rate", 0) or 0
        pp = trend.get("prior_6mo_attrition_rate", 0) or 0
        st.markdown(f'''<div class="section-card">
            <div style="font-size:0.72rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">Trend (12mo)</div>
            <div style="font-size:1.8rem;font-weight:800;color:{rc2};letter-spacing:-0.03em;text-transform:capitalize">{dir_}</div>
            <div style="font-size:0.85rem;color:#6E6E73;margin-top:8px">
                Prior 6mo: {pp:.1%}<br>Recent 6mo: <b style="color:#1D1D1F">{rn:.1%}</b>
            </div>
        </div>''', unsafe_allow_html=True)
    with g2:
        pk = seas.get("peak_month_name", "—")
        amp = seas.get("amplitude", 0) or 0
        has_seas = seas.get("has_seasonal_pattern", False)
        sc = C["orange"] if has_seas else C["text2"]
        st.markdown(f'''<div class="section-card">
            <div style="font-size:0.72rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">Seasonality</div>
            <div style="font-size:1.8rem;font-weight:800;color:{sc};letter-spacing:-0.03em">Peak: {pk}</div>
            <div style="font-size:0.85rem;color:#6E6E73;margin-top:8px">
                Amplitude: {amp:.1f} departures/mo
                {' — seasonal pattern detected' if has_seas else ''}
            </div>
        </div>''', unsafe_allow_html=True)
    with g3:
        mc = market.get("classification", "—").replace("_", " ").title()
        bls_d = market.get("bls_quarterly_delta", 0) or 0
        mk_color = (C["red"] if "headwind" in str(market.get("classification", ""))
                    else C["green_dark"] if "tailwind" in str(market.get("classification", ""))
                    else C["text2"])
        st.markdown(f'''<div class="section-card">
            <div style="font-size:0.72rem;font-weight:600;color:#86868B;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px">Market Competition</div>
            <div style="font-size:1.8rem;font-weight:800;color:{mk_color};letter-spacing:-0.03em">{mc}</div>
            <div style="font-size:0.85rem;color:#6E6E73;margin-top:8px">
                BLS Δ: {bls_d:+.1f} · Hiring difficulty: {market.get('hiring_difficulty', '?')}
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # ---- Row 6: Lead indicators + Confidence ----
    g1, g2 = st.columns([1, 1], gap="large")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Lead Indicators (3-month momentum)")
        sat_d = leads.get("satisfaction_3mo_momentum", 0)
        eng_d = leads.get("engagement_3mo_momentum", 0)
        ew = leads.get("early_warning", False)
        def _lead_row(label, delta):
            color = C["red"] if delta <= -0.10 else C["green_dark"] if delta >= 0.10 else C["text2"]
            arrow = "↓" if delta <= -0.05 else "↑" if delta >= 0.05 else "→"
            return (f'<div style="display:flex;justify-content:space-between;'
                    f'padding:8px 0;border-bottom:1px solid #F0F0F5">'
                    f'<span style="color:#1D1D1F;font-weight:500">{label}</span>'
                    f'<span style="color:{color};font-weight:700">{arrow} {delta:+.2f}</span>'
                    f'</div>')
        st.markdown(_lead_row("Satisfaction momentum", sat_d), unsafe_allow_html=True)
        st.markdown(_lead_row("Engagement momentum", eng_d), unsafe_allow_html=True)
        if ew:
            st.markdown(
                '<div style="margin-top:12px;padding:10px 14px;background:#FFF8EC;'
                'border-radius:10px;font-size:0.85rem;color:#C93400;font-weight:600">'
                '⚠ Early warning: morale declining ahead of the attrition signal.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Analysis Confidence")
        lvl = confidence.get("level", "medium")
        sc = confidence.get("score", 0)
        cc = (C["green_dark"] if lvl == "high"
              else C["orange"] if lvl == "medium"
              else C["red"])
        st.markdown(f'''<div style="display:flex;align-items:center;gap:16px;
            padding:10px 0;margin-bottom:10px">
            <div style="font-size:2.2rem;font-weight:800;color:{cc};letter-spacing:-0.03em;
            text-transform:capitalize">{lvl}</div>
            <div style="font-size:0.85rem;color:#6E6E73">
                Score {sc:.2f}<br>Based on sample size, history depth, driver concentration
            </div>
        </div>''', unsafe_allow_html=True)
        for r in confidence.get("reasons", []) or []:
            st.markdown(
                f'<div style="padding:6px 0;font-size:0.82rem;color:#6E6E73;'
                f'line-height:1.5;border-bottom:1px solid #F0F0F5">• {r}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ─── MARQUEE TICKER ─────────────────────────────────────────────────
def _render_marquee(monthly, individual):
    """Render a scrolling alert ticker with live workforce signals."""
    # Build alert items from data
    latest = monthly["month"].max()
    cur = monthly[monthly["month"] == latest]
    att_rate = individual["attrition"].mean()

    items = []
    # Top risk departments
    for _, row in cur.iterrows():
        dept = row["department"]
        dept_att = individual[individual["department"] == dept]["attrition"].mean()
        if dept_att > 0.03:
            items.append(("red", f"{dept}: {dept_att:.1%} attrition rate"))
        elif dept_att > 0.02:
            items.append(("orange", f"{dept}: {dept_att:.1%} attrition"))

    # Overall stats
    hc = int(cur["headcount"].sum())
    items.append(("blue", f"Total headcount: {hc:,}"))
    items.append(("green", f"Avg satisfaction: {cur['avg_satisfaction'].mean():.1f}/5.0"))
    items.append(("teal", f"BLS energy index: {cur['bls_energy_index'].iloc[0]:.1f}"))
    items.append(("purple", f"Active departments: {len(cur['department'].unique())}"))
    items.append(("blue", f"Overall attrition: {att_rate:.1%}"))

    color_map = {"red": "#FF3B30", "orange": "#FF9500", "blue": "#2997FF",
                 "green": "#34C759", "teal": "#5AC8FA", "purple": "#AF52DE"}

    # Duplicate items for seamless loop
    item_html = ""
    for color, text in items:
        hex_c = color_map.get(color, "#86868B")
        item_html += f'<span class="marquee-item"><span class="marquee-dot" style="background:{hex_c}"></span>{text}</span><span class="marquee-sep">&bull;</span>'

    st.markdown(f'''<div class="marquee-container">
        <div class="marquee-track">{item_html}{item_html}</div>
    </div>''', unsafe_allow_html=True)


# ─── UNIFIED PAGE ───────────────────────────────────────────────────
def render_unified(employees, monthly, individual):
    st.markdown('<h1><span class="gradient-title">Executive Dashboard</span></h1>', unsafe_allow_html=True)
    st.markdown("Real-time workforce intelligence combining talent analytics and attrition forecasting.")
    st.markdown("")

    # Top KPI row
    latest = monthly["month"].max()
    cur = monthly[monthly["month"] == latest]
    hc = int(cur["headcount"].sum())
    sat = cur["avg_satisfaction"].mean()
    att = individual["attrition"].mean()
    eng = cur["avg_engagement"].mean()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Headcount</div>
            <div class="hero-value">{hc:,}</div>
            <div class="hero-sub">Across all departments</div></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_green']}"></div>
            <div class="hero-label" style="color:{C['green_dark']}">Satisfaction</div>
            <div class="hero-value">{sat:.1f}</div>
            <div class="hero-sub">Out of 5.0</div></div>''', unsafe_allow_html=True)
    with c3:
        ac = C["red"] if att > 0.03 else C["green_dark"]
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_red']}"></div>
            <div class="hero-label" style="color:{C['red']}">Attrition</div>
            <div class="hero-value" style="color:{ac}">{att:.1%}</div>
            <div class="hero-sub">Monthly turnover rate</div></div>''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Engagement</div>
            <div class="hero-value">{eng:.1f}</div>
            <div class="hero-sub">Out of 5.0</div></div>''', unsafe_allow_html=True)

    st.markdown("")

    # Marquee alert ticker
    _render_marquee(monthly, individual)

    # Two columns: Talent + Risk
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Top Candidates")
        from data.sample_resumes import get_active_resumes, get_active_jds
        SAMPLE_RESUMES = get_active_resumes()
        SAMPLE_JOB_DESCRIPTIONS = get_active_jds()
        sbert = load_sbert()
        if sbert:
            job = SAMPLE_JOB_DESCRIPTIONS[0]
            rankings = sbert.rank_candidates(SAMPLE_RESUMES, job)
            st.markdown(f'<div style="color:#86868B;font-size:0.8rem;margin-bottom:12px">For: {job["title"]}</div>',
                        unsafe_allow_html=True)
            for i, r in enumerate(rankings[:5], 1):
                sc = score_color(r["match_score"])
                pct = r["match_score"] * 100
                st.markdown(f'''<div class="mrow">
                    <div class="mrow-rank">#{i}</div>
                    <div><div class="mrow-name">{r["candidate_name"]}</div>
                    <div class="mrow-tier">{r["match_tier"]}</div></div>
                    <div class="mrow-score" style="color:{sc}">{pct:.0f}%</div>
                </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Attrition Risk")
        forecast = load_forecast()
        if forecast:
            from tools.forecast_tools import set_forecast_engine, set_workforce_data
            set_forecast_engine(forecast); set_workforce_data(monthly, individual)
            preds = forecast.predict_all_departments(monthly)
            for p in preds[:5]:
                prob = p["attrition_probability"] * 100
                bc = C["red"] if p["risk_level"]=="CRITICAL" else C["orange"] if p["risk_level"]=="HIGH" else C["green_dark"]
                glow_cls = "drow-critical" if p["risk_level"] == "CRITICAL" else "drow-high" if p["risk_level"] == "HIGH" else ""
                ring = _svg_ring(prob / 100, bc, 32)
                st.markdown(f'''<div class="drow {glow_cls}">
                    <div><div class="drow-name">{p["department"]}</div>
                    <div class="drow-meta">{p["current_headcount"]} &rarr; {p["projected_headcount_6m"]}</div></div>
                    <div style="display:flex;align-items:center;gap:12px">
                        <div class="ring-container">{ring}</div>
                        <div style="font-weight:700;color:{bc}">{prob:.0f}%</div>
                        {rbadge(p["risk_level"])}
                    </div></div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Bottom charts
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        thc = monthly.groupby("month")["headcount"].sum().reset_index()
        fig = go.Figure(go.Scatter(x=thc["month"],y=thc["headcount"],mode="lines",
            line=dict(color=C["blue"],width=3), fill="tozeroy",
            fillcolor="rgba(0,113,227,0.06)",
            hovertemplate="Month %{x}<br>Headcount: %{y}<extra></extra>"))
        fig.update_layout(title="Total Headcount Trend")
        style_fig(fig, 320)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        bls = monthly.groupby("month")["bls_energy_index"].first().reset_index()
        fig = go.Figure(go.Scatter(x=bls["month"],y=bls["bls_energy_index"],mode="lines",
            line=dict(color=C["teal"],width=3), fill="tozeroy",
            fillcolor="rgba(90,200,250,0.06)",
            hovertemplate="Month %{x}<br>Index: %{y:.1f}<extra></extra>"))
        fig.update_layout(title="BLS Energy Employment Index")
        style_fig(fig, 320)
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)


# ─── BOARDROOM BRIEF PAGE — Proprietary Agent Brain (LLM-free) ─────
def _verdict_badge_style(verdict: str) -> str:
    """Return background + text color for a verdict badge."""
    m = {
        "FAST_TRACK":               ("#248A3D", "#fff"),
        "HIRE":                     ("#0071E3", "#fff"),
        "HIRE_WITH_ONBOARDING_FOCUS":("#5AC8FA", "#fff"),
        "HIRE_WITH_RETENTION_PLAN": ("#AF52DE", "#fff"),
        "HIRE_WITH_AGGRESSIVE_RETENTION": ("#C93400", "#fff"),
        "CONDITIONAL_HIRE":         ("#FF9500", "#fff"),
        "ADVANCE_TO_INTERVIEW":     ("#0071E3", "#fff"),
        "ADVANCE_WITH_CAVEATS":     ("#FF9500", "#fff"),
        "ADVANCE_WITH_SKILL_PLAN":  ("#5856D6", "#fff"),
        "DEFER":                    ("#6E6E73", "#fff"),
        "DECLINE":                  ("#D70015", "#fff"),
        "INCONCLUSIVE":             ("#86868B", "#fff"),
    }
    bg, fg = m.get(verdict, ("#6E6E73", "#fff"))
    return f"background:{bg};color:{fg}"


def _render_ranked_memo(memo, show_trace=True):
    """Visual renderer for triage / shortlist memos.

    These memos carry a structured ``rankings`` list in
    ``section[0].metrics["rankings"]``. Render as a horizontal bar chart
    + styled mrow cards instead of the raw markdown table that the brain
    emits for CLI / markdown rendering.
    """
    is_triage = memo.memo_type == "triage"
    name_field = "title" if is_triage else "name"
    item_label = "Role" if is_triage else "Candidate"

    section_1 = memo.sections[0] if memo.sections else None
    rankings = (
        (section_1.metrics.get("rankings") if section_1 and section_1.metrics
         else None) or []
    )

    # Headline card
    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if rankings:
        title = section_1.title if section_1 else f"{item_label} Ranking"
        st.markdown(f"### {title}")

        col_left, col_right = st.columns([1.3, 1], gap="large")
        with col_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            for i, r in enumerate(rankings, 1):
                fs = r.get("fit_score", 0) or 0
                tier_raw = r.get("recommendation_tier", "?") or "?"
                tier_label = tier_raw.replace("_", " ")
                name = r.get(name_field, "?")
                tg = r.get("top_gap") or {}
                tm = r.get("top_match") or {}
                meta_parts = [tier_label]
                if tm.get("requirement"):
                    meta_parts.append(f"matches: {tm['requirement']}")
                if tg.get("requirement"):
                    meta_parts.append(f"top gap: {tg['requirement']}")
                meta = " &middot; ".join(meta_parts)
                color = _fit_color(fs)
                st.markdown(
                    f'<div class="mrow">'
                    f'<div class="mrow-rank">#{i}</div>'
                    f'<div style="flex:1"><div class="mrow-name">{name}</div>'
                    f'<div class="mrow-tier">{meta}</div></div>'
                    f'<div class="mrow-score" style="color:{color}">{fs:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            df = pd.DataFrame(rankings)
            fig = go.Figure(go.Bar(
                y=df[name_field][::-1],
                x=df["fit_score"][::-1],
                orientation="h",
                marker=dict(
                    color=[_fit_color(s) for s in df["fit_score"][::-1]],
                    cornerradius=6, line=dict(width=0),
                ),
                text=[f"{s:.0f}" for s in df["fit_score"][::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Fit: %{x:.1f}/100<extra></extra>",
            ))
            fig.add_vline(x=70, line_dash="dot", line_color="#D2D2D7",
                          annotation=dict(text="Hire",
                                          font=dict(color="#86868B", size=10)))
            fig.add_vline(x=55, line_dash="dot", line_color="#E8E8ED",
                          annotation=dict(text="Interview",
                                          font=dict(color="#86868B", size=10)))
            fig.update_layout(
                title="Fit Score Distribution",
                xaxis=dict(range=[0, 105], title="Composite fit (0-100)"),
                yaxis=dict(title=""),
            )
            style_fig(fig, max(280, 60 + 38 * len(rankings)))
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    # Remaining sections (Recommendation, etc.) keep the standard memo styling.
    cls_options = ["memo-section-b", "memo-section-c", "memo-section-a"]
    for i, s in enumerate(memo.sections[1:]):
        body = s.narrative.replace("\n", "<br>")
        cls = cls_options[i % len(cls_options)]
        st.markdown(
            f'<div class="agent-memo" style="padding:20px 24px">'
            f'<div class="memo-section {cls}" style="margin:0">'
            f'<div class="memo-section-title">{s.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Footer
    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )
    st.markdown(
        '<div class="agent-memo" style="padding:14px 22px;margin-top:8px">'
        '<div class="memo-footer" style="margin-top:0;padding-top:0;border-top:none">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def _render_retention_plan_memo(memo, show_trace=True):
    """Visual renderer for retention_plan memos.

    Reads structured ``per_department_rollup`` and ``portfolio_top10`` from
    section metrics (set in :func:`brain._render_plan_memo`). Replaces the
    two markdown tables with hero KPIs + chart + styled action cards.
    """
    sec_summary = memo.sections[0] if len(memo.sections) > 0 else None
    sec_dept = memo.sections[1] if len(memo.sections) > 1 else None
    sec_actions = memo.sections[2] if len(memo.sections) > 2 else None

    summary_metrics = (sec_summary.metrics if sec_summary
                        and sec_summary.metrics else {}) or {}
    per_dept = ((sec_dept.metrics or {}).get("per_department_rollup")
                if sec_dept else []) or []
    portfolio = ((sec_actions.metrics or {}).get("portfolio_top10")
                  if sec_actions else []) or []

    # Headline card
    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Hero KPI row ----
    budget = summary_metrics.get("budget_usd", 0) or 0
    spent = summary_metrics.get("spent_usd", 0) or 0
    unused = summary_metrics.get("unused_budget_usd", 0) or 0
    total_red = summary_metrics.get("total_reduction_pp", 0) or 0
    avg_roi = summary_metrics.get("avg_roi_pp_per_million", 0) or 0
    n_iv = summary_metrics.get("n_interventions_selected", 0) or 0
    util_pct = (spent / budget * 100) if budget else 0

    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Budget Allocated</div>
            <div class="hero-value">${spent/1000:.0f}K</div>
            <div class="hero-sub">{util_pct:.0f}% of ${budget/1000:.0f}K — ${unused:,} unused</div>
        </div>''', unsafe_allow_html=True)
    with h2:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_green']}"></div>
            <div class="hero-label" style="color:{C['green_dark']}">Total Reduction</div>
            <div class="hero-value" style="color:{C['green_dark']}">-{total_red:.1f}<span style="font-size:1.2rem;color:#86868B">pp</span></div>
            <div class="hero-sub">Combined attrition-probability drop</div>
        </div>''', unsafe_allow_html=True)
    with h3:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Avg ROI</div>
            <div class="hero-value">{avg_roi:.0f}<span style="font-size:1.1rem;color:#86868B"> pp/$M</span></div>
            <div class="hero-sub">Pp reduction per million spent</div>
        </div>''', unsafe_allow_html=True)
    with h4:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
            <div class="hero-label" style="color:{C['orange_dark']}">Interventions</div>
            <div class="hero-value">{n_iv}</div>
            <div class="hero-sub">Across {len(per_dept)} department(s)</div>
        </div>''', unsafe_allow_html=True)

    # ---- Plan Summary narrative ----
    if sec_summary and sec_summary.narrative:
        st.markdown(
            f'<div class="agent-memo" style="padding:18px 24px;margin-top:8px">'
            f'<div class="memo-section memo-section-a" style="margin:0">'
            f'<div class="memo-section-title">{sec_summary.title}</div>'
            f'<div class="memo-section-body">{sec_summary.narrative}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Per-Department Rollup ----
    if per_dept:
        st.markdown(f"### {sec_dept.title if sec_dept else 'Per-Department Rollup'}")
        col_l, col_r = st.columns([1, 1.1], gap="large")
        with col_l:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            for rd in per_dept:
                dept = rd.get("department", "?")
                n_act = rd.get("n_actions", 0)
                cost = rd.get("cost_usd", 0) or 0
                red = rd.get("reduction_pp", 0) or 0
                color = DEPT_COLORS.get(dept, C["blue"])
                st.markdown(
                    f'<div class="drow" style="border-left:3px solid {color}">'
                    f'<div><div class="drow-name">{dept}</div>'
                    f'<div class="drow-meta">{n_act} action(s) &middot; ${cost:,}</div></div>'
                    f'<div style="font-weight:800;color:{C["green_dark"]};font-size:1.15rem;letter-spacing:-0.02em">'
                    f'-{red:.1f}<span style="font-size:0.78rem;color:#86868B">pp</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            df = pd.DataFrame(per_dept)
            depts_sorted = list(df["department"][::-1])
            reds_sorted = list(df["reduction_pp"][::-1])
            costs_sorted = list(df["cost_usd"][::-1])
            colors = [DEPT_COLORS.get(d, C["blue"]) for d in depts_sorted]
            fig = go.Figure(go.Bar(
                y=depts_sorted, x=reds_sorted, orientation="h",
                marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
                text=[f"-{r:.1f}pp" for r in reds_sorted],
                textposition="outside",
                customdata=[[c] for c in costs_sorted],
                hovertemplate="<b>%{y}</b><br>Reduction: -%{x:.1f}pp"
                              "<br>Cost: $%{customdata[0]:,.0f}<extra></extra>",
            ))
            fig.update_layout(
                title="Reduction by Department",
                xaxis=dict(title="Attrition reduction (pp)"),
                yaxis=dict(title=""),
            )
            style_fig(fig, max(280, 60 + 38 * len(per_dept)))
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Top 10 Actions ----
    if portfolio:
        st.markdown(f"### {sec_actions.title if sec_actions else 'Top 10 Actions (by ROI)'}")
        col_l, col_r = st.columns([1.2, 1], gap="large")
        with col_l:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            for i, a in enumerate(portfolio, 1):
                dept = a.get("department", "?")
                label = a.get("intervention_label", "?")
                mag = a.get("magnitude", 0) or 0
                cost = a.get("cost_usd", 0) or 0
                red = a.get("attrition_reduction_pp", 0) or 0
                roi = a.get("roi_pp_per_million", 0) or 0
                lead = a.get("lead_time_days")
                color = DEPT_COLORS.get(dept, C["purple"])
                meta_parts = [f"mag {mag:.2f}", f"${cost:,}", f"-{red:.1f}pp"]
                if lead is not None:
                    meta_parts.append(f"{lead}d lead")
                meta = " &middot; ".join(meta_parts)
                st.markdown(
                    f'<div class="mrow">'
                    f'<div class="mrow-rank">#{i}</div>'
                    f'<div style="flex:1">'
                    f'<div class="mrow-name">{label} <span style="color:{color};font-weight:500">&rarr; {dept}</span></div>'
                    f'<div class="mrow-tier">{meta}</div></div>'
                    f'<div class="mrow-score" style="color:{C["purple"]}">{roi:.0f}<span style="font-size:0.78rem;color:#86868B"> pp/$M</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            labels = [f"#{i} {a['intervention_label']} → {a['department']}"
                      for i, a in enumerate(portfolio, 1)][::-1]
            rois = [a.get("roi_pp_per_million", 0) or 0
                    for a in portfolio][::-1]
            colors = [DEPT_COLORS.get(a["department"], C["purple"])
                      for a in portfolio][::-1]
            fig = go.Figure(go.Bar(
                y=labels, x=rois, orientation="h",
                marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
                text=[f"{r:.0f}" for r in rois],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>ROI: %{x:.0f} pp/$M<extra></extra>",
            ))
            fig.update_layout(
                title="ROI per Action (pp / $M)",
                xaxis=dict(title="ROI"),
                yaxis=dict(title="", tickfont=dict(size=10)),
            )
            style_fig(fig, max(320, 60 + 38 * len(portfolio)))
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    # Render any extra sections beyond the expected 3
    cls_options = ["memo-section-a", "memo-section-b", "memo-section-c"]
    for i, s in enumerate(memo.sections[3:]):
        body = s.narrative.replace("\n", "<br>")
        cls = cls_options[i % len(cls_options)]
        st.markdown(
            f'<div class="agent-memo" style="padding:20px 24px">'
            f'<div class="memo-section {cls}" style="margin:0">'
            f'<div class="memo-section-title">{s.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Footer
    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )
    st.markdown(
        '<div class="agent-memo" style="padding:14px 22px;margin-top:8px">'
        '<div class="memo-footer" style="margin-top:0;padding-top:0;border-top:none">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def _render_risk_scan_memo(memo, show_trace=True):
    """Visual renderer for risk_scan memos.

    Reads structured ``ranked``, ``deep_dive`` and ``actions`` lists from
    section metrics (set in :func:`brain._render_scan_memo`). Replaces the
    ranking table + markdown narrative with hero KPIs + chart + cards.
    """
    sec_rank = memo.sections[0] if len(memo.sections) > 0 else None
    sec_dive = memo.sections[1] if len(memo.sections) > 1 else None
    sec_act = memo.sections[2] if len(memo.sections) > 2 else None

    rank_metrics = (sec_rank.metrics or {}) if sec_rank else {}
    ranked = rank_metrics.get("ranked", []) or []
    deep_dive = ((sec_dive.metrics or {}).get("deep_dive")
                  if sec_dive else []) or []
    act_metrics = (sec_act.metrics or {}) if sec_act else {}
    actions = act_metrics.get("actions", []) or []

    # Headline card
    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Hero KPIs ----
    n_dept = rank_metrics.get("n_departments", len(ranked))
    n_crit = rank_metrics.get("n_critical", 0)
    n_high = rank_metrics.get("n_high", 0)
    total_exposure = sum(d.get("replacement_cost_usd", 0) or 0
                          for d in deep_dive)
    total_action_cost = act_metrics.get("total_cost_usd", 0) or 0

    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Departments Scanned</div>
            <div class="hero-value">{n_dept}</div>
            <div class="hero-sub">Across the org</div>
        </div>''', unsafe_allow_html=True)
    with h2:
        crit_color = C["red"] if n_crit else C["text2"]
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_red']}"></div>
            <div class="hero-label" style="color:{C['red']}">Critical</div>
            <div class="hero-value" style="color:{crit_color}">{n_crit}</div>
            <div class="hero-sub">Department(s) at CRITICAL risk</div>
        </div>''', unsafe_allow_html=True)
    with h3:
        high_color = C["orange_dark"] if n_high else C["text2"]
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
            <div class="hero-label" style="color:{C['orange']}">High</div>
            <div class="hero-value" style="color:{high_color}">{n_high}</div>
            <div class="hero-sub">Department(s) at HIGH risk</div>
        </div>''', unsafe_allow_html=True)
    with h4:
        if total_exposure >= 1_000_000:
            disp = f"${total_exposure/1_000_000:.1f}M"
        elif total_exposure >= 1000:
            disp = f"${total_exposure/1000:.0f}K"
        else:
            disp = f"${total_exposure:,}"
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Top-3 Exposure</div>
            <div class="hero-value" style="color:{C['orange_dark']}">{disp}</div>
            <div class="hero-sub">Modelled replacement cost</div>
        </div>''', unsafe_allow_html=True)

    # ---- Department Risk Ranking ----
    if ranked:
        st.markdown(f"### {sec_rank.title if sec_rank else 'Department Risk Ranking'}")
        col_l, col_r = st.columns([1.2, 1], gap="large")
        with col_l:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            for i, r in enumerate(ranked, 1):
                dept = r.get("department", "?")
                rl = r.get("risk_level", "LOW")
                prob = (r.get("attrition_probability", 0) or 0) * 100
                cur_hc = r.get("current_headcount", 0) or 0
                hc_6m = r.get("projected_headcount_6m", cur_hc) or cur_hc
                bc = (C["red"] if rl == "CRITICAL"
                      else C["orange"] if rl == "HIGH"
                      else C["green_dark"])
                glow_cls = ("drow-critical" if rl == "CRITICAL"
                            else "drow-high" if rl == "HIGH" else "")
                ring = _svg_ring(prob / 100, bc, 36)
                st.markdown(f'''<div class="drow {glow_cls}">
                    <div style="display:flex;align-items:center;gap:10px">
                        <span style="color:#86868B;font-weight:700;min-width:24px">#{i}</span>
                        <div><div class="drow-name">{dept}</div>
                        <div class="drow-meta">HC: {cur_hc} &rarr; {hc_6m} (6-mo)</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:14px">
                        <div class="ring-container">{ring}</div>
                        <div style="font-weight:700;font-size:0.9rem;color:{bc};min-width:42px;text-align:right">{prob:.0f}%</div>
                        {rbadge(rl)}
                    </div>
                </div>''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            df = pd.DataFrame(ranked)
            depts_sorted = list(df["department"][::-1])
            probs_sorted = [(p or 0) * 100 for p in df["attrition_probability"][::-1]]
            risks_sorted = list(df["risk_level"][::-1])
            colors = [(C["red"] if rl == "CRITICAL"
                        else C["orange"] if rl == "HIGH"
                        else C["green_dark"]) for rl in risks_sorted]
            fig = go.Figure(go.Bar(
                y=depts_sorted, x=probs_sorted, orientation="h",
                marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
                text=[f"{p:.0f}%" for p in probs_sorted],
                textposition="outside",
                customdata=[[rl] for rl in risks_sorted],
                hovertemplate="<b>%{y}</b><br>Attrition prob: %{x:.1f}%"
                              "<br>Risk: %{customdata[0]}<extra></extra>",
            ))
            fig.add_vline(x=70, line_dash="dot", line_color="#FF3B30",
                          annotation=dict(text="Critical",
                                          font=dict(color="#86868B", size=10)))
            fig.add_vline(x=50, line_dash="dot", line_color="#FF9500",
                          annotation=dict(text="High",
                                          font=dict(color="#86868B", size=10)))
            fig.update_layout(
                title="Attrition Probability",
                xaxis=dict(range=[0, 105], title="% probability (12-mo)"),
                yaxis=dict(title=""),
            )
            style_fig(fig, max(280, 60 + 38 * len(ranked)))
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Threshold Sensitivity (multi-policy classification) ----
    policy_class = rank_metrics.get("policy_classifications", []) or []
    policy_thresholds = rank_metrics.get("policy_thresholds", {}) or {}
    policy_blurbs = rank_metrics.get("policy_blurbs", {}) or {}
    n_disagree = rank_metrics.get("n_policy_disagreements", 0) or 0
    n_jumps = rank_metrics.get("n_policy_jumps", 0) or 0
    if policy_class:
        st.markdown("### Threshold Sensitivity — Multi-Brain Risk Classification")
        st.markdown(
            '<div style="color:#86868B;font-size:0.82rem;margin-bottom:14px">'
            'Each department\'s attrition probability classified under three '
            'different risk-band policies. Disagreement is itself a '
            'diagnostic — borderline departments are where the threshold '
            'choice is doing the work, and they\'re the natural candidates '
            'for human review.</div>',
            unsafe_allow_html=True,
        )

        # Hero strip — agreement / disagreement / jump counts.
        n_total = len(policy_class)
        n_agree = n_total - n_disagree
        agree_pct = (n_agree / n_total * 100) if n_total else 0
        ts1, ts2, ts3, ts4 = st.columns(4, gap="medium")
        with ts1:
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
                <div class="hero-label" style="color:{C['blue']}">Departments</div>
                <div class="hero-value">{n_total}</div>
                <div class="hero-sub">Classified across 3 policies</div>
            </div>''', unsafe_allow_html=True)
        with ts2:
            ag_color = C["green_dark"] if agree_pct >= 75 else C["orange"]
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_green']}"></div>
                <div class="hero-label" style="color:{C['green_dark']}">All-3 Agree</div>
                <div class="hero-value" style="color:{ag_color}">{n_agree}<span style="font-size:1.0rem;color:#86868B"> / {n_total}</span></div>
                <div class="hero-sub">{agree_pct:.0f}% policy-stable</div>
            </div>''', unsafe_allow_html=True)
        with ts3:
            d_color = C["orange_dark"] if n_disagree else C["text2"]
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_orange']}"></div>
                <div class="hero-label" style="color:{C['orange']}">Borderline</div>
                <div class="hero-value" style="color:{d_color}">{n_disagree}</div>
                <div class="hero-sub">≥1 policy disagrees</div>
            </div>''', unsafe_allow_html=True)
        with ts4:
            j_color = C["red"] if n_jumps else C["text2"]
            st.markdown(f'''<div class="hero-card">
                <div class="hero-accent" style="background:{C['gradient_red']}"></div>
                <div class="hero-label" style="color:{C['red']}">Cross-Band Jumps</div>
                <div class="hero-value" style="color:{j_color}">{n_jumps}</div>
                <div class="hero-sub">≥2-band spread (escalate)</div>
            </div>''', unsafe_allow_html=True)

        # Policy-cutoff legend
        legend_parts = []
        for key in ("sophisticated", "conservative", "heuristic"):
            blurb = policy_blurbs.get(key, "")
            policy_label = {"sophisticated": "Sophisticated",
                            "conservative":  "Conservative",
                            "heuristic":     "Heuristic baseline"}.get(key, key)
            color = {"sophisticated": C["blue"],
                     "conservative":  C["orange"],
                     "heuristic":     C["purple"]}.get(key, C["text2"])
            legend_parts.append(
                f'<div style="display:flex;gap:10px;align-items:flex-start;'
                f'padding:6px 0">'
                f'<div style="min-width:8px;width:8px;height:14px;background:{color};'
                f'border-radius:2px;margin-top:4px"></div>'
                f'<div><strong style="color:#1D1D1F;font-size:0.85rem">{policy_label}</strong>'
                f'<div style="color:#6E6E73;font-size:0.78rem;line-height:1.45">{blurb}</div></div>'
                f'</div>'
            )
        st.markdown(
            '<div class="section-card" style="padding:14px 18px;margin-bottom:14px">'
            '<div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;'
            'letter-spacing:0.06em;font-weight:600;margin-bottom:8px">Policies</div>'
            + "".join(legend_parts)
            + '</div>',
            unsafe_allow_html=True,
        )

        # Per-department comparison: probability bar with cutoff lines + 3-policy badges.
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;'
            'letter-spacing:0.06em;font-weight:600;margin-bottom:10px">'
            'Per-department classifications (sorted by probability)</div>',
            unsafe_allow_html=True,
        )
        # Sort by probability descending so the borderline cluster is at the top.
        policy_class_sorted = sorted(
            policy_class,
            key=lambda d: -(d.get("attrition_probability") or 0),
        )

        def _risk_color(level):
            return {"CRITICAL": C["red"], "HIGH": C["orange"],
                    "MEDIUM": C["purple"], "LOW": C["green_dark"]}.get(level, C["text2"])

        for d in policy_class_sorted:
            dept = d.get("department", "?")
            prob = (d.get("attrition_probability") or 0) * 100
            cls = d.get("policy_classifications") or {}
            soph = cls.get("sophisticated", "?")
            cons = cls.get("conservative", "?")
            heur = cls.get("heuristic", "?")
            agree = d.get("agreement", False)
            spread = d.get("severity_spread", 0)
            border = (C["green_dark"] if agree
                      else C["orange"] if spread == 1
                      else C["red"])
            agree_icon = ("✓" if agree
                          else "△" if spread == 1
                          else "⚠")
            agree_label = ("policy-stable" if agree
                            else f"adjacent disagreement"
                            if spread == 1
                            else f"cross-band jump")

            # Cutoff line markers from the absolute policies (over a 0-100 bar).
            soph_crit, soph_high, soph_med = (policy_thresholds
                                                 .get("sophisticated", (0.7, 0.5, 0.3)))
            cons_crit, cons_high, cons_med = (policy_thresholds
                                                 .get("conservative", (0.55, 0.35, 0.20)))

            st.markdown(
                f'<div class="drow" style="border-left:3px solid {border};'
                f'flex-direction:column;align-items:stretch;gap:8px;padding:14px 18px">'
                # Top row — dept + probability + agreement badge
                f'<div style="display:flex;align-items:center;justify-content:space-between;gap:14px">'
                f'<div style="display:flex;flex-direction:column">'
                f'<span class="drow-name">{dept}</span>'
                f'<span class="drow-meta">{prob:.0f}% attrition probability '
                f'· {agree_icon} {agree_label}</span></div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">'
                # 3 mini badges, one per policy
                f'<span style="font-size:0.66rem;color:{C["blue"]};font-weight:700;'
                f'letter-spacing:0.04em;text-transform:uppercase">SOPH</span>'
                f'{rbadge(soph)}'
                f'<span style="font-size:0.66rem;color:{C["orange"]};font-weight:700;'
                f'letter-spacing:0.04em;text-transform:uppercase;margin-left:6px">CONS</span>'
                f'{rbadge(cons)}'
                f'<span style="font-size:0.66rem;color:{C["purple"]};font-weight:700;'
                f'letter-spacing:0.04em;text-transform:uppercase;margin-left:6px">HEUR</span>'
                f'{rbadge(heur)}'
                f'</div>'
                f'</div>'
                # Probability bar with all 4 cutoff markers
                f'<div style="position:relative;height:12px;background:#F0F0F5;border-radius:6px;margin-top:4px">'
                # Filled portion to the probability
                f'<div style="position:absolute;left:0;top:0;height:100%;'
                f'width:{prob:.1f}%;background:{_risk_color(soph)};opacity:0.55;'
                f'border-radius:6px"></div>'
                # Soph cutoff lines (blue)
                f'<div style="position:absolute;left:{soph_crit*100:.1f}%;top:-2px;'
                f'width:2px;height:16px;background:{C["blue"]}" title="Soph CRITICAL"></div>'
                f'<div style="position:absolute;left:{soph_high*100:.1f}%;top:-2px;'
                f'width:2px;height:16px;background:{C["blue"]};opacity:0.6"></div>'
                # Cons cutoff lines (orange)
                f'<div style="position:absolute;left:{cons_crit*100:.1f}%;top:-2px;'
                f'width:2px;height:16px;background:{C["orange"]}" title="Cons CRITICAL"></div>'
                f'<div style="position:absolute;left:{cons_high*100:.1f}%;top:-2px;'
                f'width:2px;height:16px;background:{C["orange"]};opacity:0.6"></div>'
                # Probability marker (dark vertical)
                f'<div style="position:absolute;left:calc({prob:.1f}% - 1px);top:-4px;'
                f'width:3px;height:20px;background:#1D1D1F;border-radius:1px"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size:0.72rem;color:#86868B;margin-top:12px;line-height:1.5">'
            'Vertical markers: <strong style="color:#0071E3">blue</strong> = '
            'Sophisticated cutoffs (CRITICAL / HIGH), '
            '<strong style="color:#FF9500">orange</strong> = Conservative cutoffs. '
            'Dark line = the department\'s actual probability.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Top-3 Deep-Dive ----
    if deep_dive:
        st.markdown(f"### {sec_dive.title if sec_dive else 'Top-3 Deep-Dive'}")
        cols = st.columns(len(deep_dive), gap="medium")
        for col, dd in zip(cols, deep_dive):
            with col:
                dept = dd.get("department", "?")
                rl = dd.get("risk_level", "LOW")
                prob = (dd.get("attrition_probability", 0) or 0) * 100
                std_pp = (dd.get("mc_std") or 0) * 100
                mc_lo = (dd.get("mc_p5") or 0) * 100
                mc_hi = (dd.get("mc_p95") or 0) * 100
                cf_lo = dd.get("conformal_p_low")
                cf_hi = dd.get("conformal_p_high")
                cf_w = dd.get("conformal_width")
                drv_label = dd.get("primary_driver_label") or "—"
                drv_pct = dd.get("primary_driver_pct") or 0
                rt_crit = dd.get("retirement_cliff_critical", 0) or 0
                comp_gap = dd.get("comp_gap_count", 0) or 0
                kn_sev = dd.get("knowledge_loss_severity") or "—"
                cost = dd.get("replacement_cost_usd", 0) or 0
                cost_disp = (f"${cost/1_000_000:.1f}M" if cost >= 1_000_000
                              else f"${cost/1000:.0f}K" if cost >= 1000
                              else f"${cost:,}")
                bc = (C["red"] if rl == "CRITICAL"
                      else C["orange"] if rl == "HIGH"
                      else C["green_dark"])

                conformal_html = ""
                if cf_lo is not None and cf_hi is not None:
                    conformal_html = (
                        f'<div style="font-size:0.75rem;color:#86868B;margin-top:4px">'
                        f'Conformal 90% CI [{cf_lo*100:.0f}%, {cf_hi*100:.0f}%]'
                        + (f' (width {cf_w:.2f})' if cf_w is not None else '')
                        + '</div>'
                    )

                st.markdown(f'''<div class="section-card" style="border-left:3px solid {bc}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                        <div>
                            <div style="font-weight:700;font-size:1.05rem;color:#1D1D1F">{dept}</div>
                            <div style="font-size:0.78rem;color:#86868B;margin-top:2px">Replacement exposure</div>
                            <div style="font-size:1.1rem;font-weight:700;color:{C["orange_dark"]};letter-spacing:-0.02em">{cost_disp}</div>
                        </div>
                        <div>{rbadge(rl)}</div>
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;color:{bc};letter-spacing:-0.03em;line-height:1">
                        {prob:.0f}<span style="font-size:0.95rem;color:#86868B">% &plusmn; {std_pp:.1f}pp</span>
                    </div>
                    <div style="font-size:0.75rem;color:#86868B;margin-top:6px">
                        MC-Dropout 90% CI [{mc_lo:.0f}%, {mc_hi:.0f}%]
                    </div>
                    {conformal_html}
                    <div style="margin-top:14px;padding-top:12px;border-top:1px solid #F0F0F5">
                        <div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:4px">Primary Driver</div>
                        <div style="font-size:0.92rem;color:#1D1D1F;font-weight:600">{drv_label}</div>
                        <div style="font-size:0.78rem;color:#86868B">{drv_pct:.0f}% of risk</div>
                    </div>
                    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #F0F0F5">
                        <div style="font-size:0.72rem;color:#86868B;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:6px">At-Risk Cohorts</div>
                        <span class="etag etag-employer">{rt_crit} retirement-cliff</span>
                        <span class="etag etag-skill">{comp_gap} comp-gap</span>
                        <span class="etag etag-cert">knowledge-loss: {kn_sev}</span>
                    </div>
                </div>''', unsafe_allow_html=True)

    # ---- Top Action Items ----
    if actions:
        st.markdown(f"### {sec_act.title if sec_act else 'Top Action Items'}")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        for i, a in enumerate(actions, 1):
            dept = a.get("department", "?")
            action_label = a.get("action") or "?"
            cohort = a.get("cohort_size", 0) or 0
            driver = a.get("driver", "?")
            cost = a.get("program_cost_usd", 0) or 0
            roi = a.get("roi_multiplier")
            color = DEPT_COLORS.get(dept, C["purple"])
            roi_html = (f'<div class="mrow-score" style="color:{C["purple"]}">'
                        f'{roi}<span style="font-size:0.78rem;color:#86868B">×</span>'
                        f'</div>') if roi is not None else ''
            st.markdown(
                f'<div class="mrow">'
                f'<div class="mrow-rank">#{i}</div>'
                f'<div style="flex:1">'
                f'<div class="mrow-name">{action_label} '
                f'<span style="color:{color};font-weight:500">&rarr; {dept}</span></div>'
                f'<div class="mrow-tier">{cohort} people &middot; targets {driver} '
                f'&middot; ${cost:,}</div></div>'
                f'{roi_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="color:#86868B;font-size:0.78rem;margin-top:12px">'
            f'Total program cost: <strong style="color:#1D1D1F">'
            f'${total_action_cost:,}</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    elif sec_act:
        body = (sec_act.narrative or "").replace("\n", "<br>")
        st.markdown(
            f'<div class="agent-memo" style="padding:18px 24px">'
            f'<div class="memo-section memo-section-c" style="margin:0">'
            f'<div class="memo-section-title">{sec_act.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Render any extra sections beyond the expected 3
    cls_options = ["memo-section-a", "memo-section-b", "memo-section-c"]
    for i, s in enumerate(memo.sections[3:]):
        body = s.narrative.replace("\n", "<br>")
        cls = cls_options[i % len(cls_options)]
        st.markdown(
            f'<div class="agent-memo" style="padding:20px 24px">'
            f'<div class="memo-section {cls}" style="margin:0">'
            f'<div class="memo-section-title">{s.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Footer
    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )
    st.markdown(
        '<div class="agent-memo" style="padding:14px 22px;margin-top:8px">'
        '<div class="memo-footer" style="margin-top:0;padding-top:0;border-top:none">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def _render_consensus_decision_memo(memo, show_trace=True):
    """Visual renderer for multi-brain consensus_decision memos.

    Reads structured ``verdicts_full`` from section[0].metrics (set in
    :func:`brain._render_consensus_memo`) and renders 3 side-by-side brain
    comparison cards plus a consensus banner and action recommendation —
    instead of the markdown table that the brain emits.
    """
    sec_verdicts = memo.sections[0] if len(memo.sections) > 0 else None
    sec_consensus = memo.sections[1] if len(memo.sections) > 1 else None
    sec_reference = memo.sections[2] if len(memo.sections) > 2 else None

    verdicts_full = ((sec_verdicts.metrics or {}).get("verdicts_full")
                      if sec_verdicts else []) or []
    consensus_metrics = (sec_consensus.metrics or {}) if sec_consensus else {}
    consensus_class = (consensus_metrics.get("consensus_class")
                        or (memo.metrics or {}).get("consensus_class")
                        or "STRONG_CONSENSUS")

    consensus_color = {
        "STRONG_CONSENSUS":   ("#34C759", "#248A3D"),  # green
        "MAJORITY_CONSENSUS": ("#FF9500", "#C93400"),  # orange
        "NO_CONSENSUS":       ("#FF3B30", "#D70015"),  # red
    }.get(consensus_class, ("#86868B", "#6E6E73"))
    bg_c, fg_c = consensus_color

    consensus_label = {
        "STRONG_CONSENSUS":   "✓ All 3 agree",
        "MAJORITY_CONSENSUS": "⚠ 2/3 majority",
        "NO_CONSENSUS":       "⛔ NO CONSENSUS",
    }.get(consensus_class, consensus_class)

    # Headline card with consensus banner
    verdict_html = ""
    if memo.verdict:
        vstyle = _verdict_badge_style(memo.verdict)
        verdict_html = (
            f'<div class="memo-verdict-row">'
            f'<span class="memo-verdict-badge" style="{vstyle}">{memo.verdict}</span>'
            + (f'<span class="memo-rule-badge">{memo.rule_applied}</span>'
               if memo.rule_applied else "")
            + '</div>'
        )
    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        + verdict_html
        + f'<div style="margin-top:12px"><span style="background:{bg_c};'
          f'color:#fff;padding:6px 16px;border-radius:18px;font-size:0.82rem;'
          f'font-weight:700;letter-spacing:0.04em;text-transform:uppercase">'
          f'{consensus_label}</span></div>'
        + '</div>',
        unsafe_allow_html=True,
    )

    # ---- Decision Inputs strip ----
    md = memo.metrics or {}
    department = md.get("department", "—")
    fit_score = md.get("fit_score")
    fit_tier = md.get("fit_tier") or "—"
    risk_level = md.get("risk_level") or "—"
    risk_color = (C["red"] if risk_level == "CRITICAL"
                  else C["orange"] if risk_level == "HIGH"
                  else C["green_dark"] if risk_level == "LOW"
                  else C["text2"])
    fit_color = _fit_color(fit_score) if isinstance(fit_score, (int, float)) else C["text2"]
    fit_disp = f"{fit_score:.0f}" if isinstance(fit_score, (int, float)) else "—"

    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_blue']}"></div>
            <div class="hero-label" style="color:{C['blue']}">Receiving Dept</div>
            <div class="hero-value" style="font-size:1.6rem">{department}</div>
            <div class="hero-sub">Decision context</div>
        </div>''', unsafe_allow_html=True)
    with h2:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_purple']}"></div>
            <div class="hero-label" style="color:{C['purple']}">Fit Score</div>
            <div class="hero-value" style="color:{fit_color}">{fit_disp}<span style="font-size:1.2rem;color:#86868B">/100</span></div>
            <div class="hero-sub">Composite candidate fit</div>
        </div>''', unsafe_allow_html=True)
    with h3:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_green']}"></div>
            <div class="hero-label" style="color:{C['green_dark']}">Fit Tier</div>
            <div class="hero-value" style="font-size:1.4rem">{fit_tier.replace("_", " ")}</div>
            <div class="hero-sub">Talent-side cell input</div>
        </div>''', unsafe_allow_html=True)
    with h4:
        st.markdown(f'''<div class="hero-card">
            <div class="hero-accent" style="background:{C['gradient_red']}"></div>
            <div class="hero-label" style="color:{C['red']}">Team Risk</div>
            <div class="hero-value" style="color:{risk_color};font-size:1.6rem">{risk_level}</div>
            <div class="hero-sub">Workforce-side cell input</div>
        </div>''', unsafe_allow_html=True)

    # ---- Three-Brain Comparison ----
    if verdicts_full:
        st.markdown(f"### {sec_verdicts.title if sec_verdicts else 'Per-Brain Verdicts'}")
        # Color per brain — distinguish them visually.
        brain_colors = {
            "Sophisticated":      C["blue"],
            "Conservative":       C["orange"],
            "Heuristic baseline": C["purple"],
        }
        cols = st.columns(len(verdicts_full), gap="medium")
        for col, v in zip(cols, verdicts_full):
            with col:
                brain_name = v.get("brain", "?")
                bcol = brain_colors.get(brain_name, C["text2"])
                verdict = v.get("verdict") or "?"
                rule = v.get("rule") or "—"
                family = v.get("family") or "?"
                rationale = v.get("rationale") or ""
                # Truncate very long rationale text — visible expander option.
                short_rationale = rationale
                if len(short_rationale) > 280:
                    short_rationale = short_rationale[:280].rstrip() + "…"
                vstyle = _verdict_badge_style(verdict)
                st.markdown(f'''<div class="section-card" style="border-top:3px solid {bcol};min-height:240px">
                    <div style="font-size:0.7rem;color:{bcol};font-weight:700;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">{brain_name}</div>
                    <div style="margin:8px 0 12px 0">
                        <span class="memo-verdict-badge" style="{vstyle}">{verdict}</span>
                    </div>
                    <div style="font-size:0.72rem;color:#86868B;margin-bottom:4px">RULE</div>
                    <div style="font-family:'SF Mono',monospace;font-size:0.78rem;color:#1D1D1F;background:#F0F0F5;padding:4px 10px;border-radius:8px;display:inline-block;margin-bottom:10px">{rule}</div>
                    <div style="font-size:0.72rem;color:#86868B;margin:6px 0 4px 0">FAMILY</div>
                    <div style="font-size:0.85rem;font-weight:600;color:{bcol};margin-bottom:14px">{family}</div>
                    <div style="font-size:0.72rem;color:#86868B;margin-bottom:4px">RATIONALE</div>
                    <div style="font-size:0.82rem;color:#1D1D1F;line-height:1.5">{short_rationale}</div>
                </div>''', unsafe_allow_html=True)
                if len(rationale) > 280:
                    with st.expander("Full rationale"):
                        st.markdown(rationale)

    # ---- Recommended Action ----
    if sec_consensus:
        action_text = sec_consensus.narrative or ""
        st.markdown(
            f'<div class="agent-memo" style="padding:20px 24px;'
            f'border-left:3px solid {bg_c}">'
            f'<div style="font-size:0.7rem;color:{fg_c};font-weight:700;'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">'
            f'Recommended Action</div>'
            f'<div style="font-size:1.0rem;font-weight:700;color:#1D1D1F;'
            f'margin-bottom:8px">{consensus_label}</div>'
            f'<div style="font-size:0.92rem;color:#1D1D1F;line-height:1.6">'
            f'{action_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- Reference memo (collapsed) ----
    if sec_reference and sec_reference.narrative:
        with st.expander("Reference memo — Sophisticated Brain (full analysis)"):
            st.markdown(sec_reference.narrative)

    # Footer
    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
        f'<span class="memo-footer-item"><strong>Brains:</strong> 3 (Sophisticated · Conservative · Heuristic)</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )
    st.markdown(
        '<div class="agent-memo" style="padding:14px 22px;margin-top:8px">'
        '<div class="memo-footer" style="margin-top:0;padding-top:0;border-top:none">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def _render_counterfactual_block(section):
    """Visual block for the joint-hire counterfactual flip explainer.

    Reads ``candidate_paths`` and ``team_paths`` from section.metrics.
    Renders styled cards + a feasibility-coded chart that compare each
    flip lever's projected delta side-by-side.
    """
    metrics = section.metrics or {}
    cand_paths = metrics.get("candidate_paths") or []
    team_paths = metrics.get("team_paths") or []
    trigger = metrics.get("trigger_verdict", "—")
    current_tier = metrics.get("current_fit_tier", "?")
    current_risk = metrics.get("current_risk_level", "?")
    flip_feasible = metrics.get("flip_feasible")

    # Section header
    st.markdown(f"### {section.title}")

    # Intro / no-flip-available banner
    if not (cand_paths or team_paths):
        st.markdown(
            f'<div class="agent-memo" style="padding:18px 22px;'
            f'border-left:3px solid {C["red"]}">'
            f'<div style="font-weight:700;color:#1D1D1F;font-size:0.95rem;'
            f'margin-bottom:6px">No single-lever flip available</div>'
            f'<div style="font-size:0.88rem;color:#1D1D1F;line-height:1.55">'
            f'No single candidate-side gap closure or team-side intervention '
            f'(at canonical magnitude) moves the cell '
            f'<strong>(fit={current_tier}, risk={current_risk})</strong> '
            f'out of <code>{trigger}</code> territory. Recommend re-scoping '
            f'the role, expanding sourcing, or deferring the req until the '
            f'receiving team stabilizes independently.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    intro = (
        f'Under the current policy this verdict is '
        f'<code style="background:#FFF0EF;color:#D70015;padding:2px 8px;'
        f'border-radius:6px">{trigger}</code> at '
        f'<strong>(fit={current_tier}, risk={current_risk})</strong>. The '
        f'levers below would shift the (fit × risk) cell into a '
        f'hire-family or advance-family verdict. Feasibility grade: '
        f'<strong style="color:#248A3D">high</strong> = causal & fast · '
        f'<strong style="color:#FF9500">medium</strong> = mixed-causal '
        f'or 3–6 months · <strong style="color:#D70015">low</strong> = '
        f'correlational-only or degree-level gap.'
    )
    st.markdown(
        f'<div style="background:#F7FBFF;border-left:3px solid {C["blue"]};'
        f'padding:12px 18px;border-radius:0 12px 12px 0;font-size:0.85rem;'
        f'color:#1D1D1F;line-height:1.5;margin-bottom:14px">{intro}</div>',
        unsafe_allow_html=True,
    )

    feas_color = {
        "high":   C["green_dark"],
        "medium": C["orange"],
        "low":    C["red"],
    }
    feas_rank = {"high": 0, "medium": 1, "low": 2}

    col_l, col_r = st.columns([1.3, 1], gap="large")

    # ---- Left: lever cards ----
    with col_l:
        if cand_paths:
            st.markdown('<div class="section-card" style="margin-bottom:12px">',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.72rem;color:{C["blue"]};font-weight:700;'
                f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">'
                f'Candidate-side flips ({len(cand_paths)})</div>',
                unsafe_allow_html=True,
            )
            for p in cand_paths:
                lever = p.get("lever") or "?"
                crit = p.get("criticality") or "STANDARD"
                ltype = p.get("lever_type") or "skill"
                delta = p.get("fit_delta_points")
                delta_str = (f"+{delta:.1f}pts" if isinstance(delta, (int, float))
                             else "+? pts")
                projected_tier = (p.get("projected_fit_tier") or "?").replace("_", " ")
                projected_verdict = p.get("projected_verdict") or "?"
                projected_rule = p.get("projected_rule") or "—"
                eta = p.get("eta_days")
                eta_str = f"~{eta}d" if eta else "—"
                feas = (p.get("feasibility") or "medium").lower()
                feas_c = feas_color.get(feas, C["text2"])
                action = p.get("action") or ""
                vstyle = _verdict_badge_style(projected_verdict)
                st.markdown(
                    f'<div class="mrow" style="border-left:3px solid {feas_c}">'
                    f'<div style="flex:1">'
                    f'<div class="mrow-name">Close <strong>{lever}</strong> '
                    f'<span class="crit-badge crit-{crit}">{crit.replace("_"," ")}</span> '
                    f'<span style="color:#86868B;font-size:0.78rem;'
                    f'font-weight:500">·&nbsp;{ltype}</span></div>'
                    f'<div class="mrow-tier">'
                    f'fit lifts <strong style="color:#1D1D1F">{delta_str}</strong> '
                    f'→ <strong>{projected_tier}</strong> · ETA {eta_str} · '
                    f'feasibility <strong style="color:{feas_c}">{feas}</strong>'
                    f'</div>'
                    f'<div style="font-size:0.78rem;color:#6E6E73;margin-top:4px">'
                    f'<span class="memo-verdict-badge" style="{vstyle};font-size:0.68rem">{projected_verdict}</span> '
                    f'<code style="font-size:0.7rem;background:#F0F0F5;padding:1px 6px;'
                    f'border-radius:6px;color:#1D1D1F">{projected_rule}</code>'
                    f'</div>'
                    + (f'<div style="font-size:0.78rem;color:#86868B;margin-top:4px;'
                       f'line-height:1.4">Action: {action}</div>' if action else '')
                    + f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        if team_paths:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.72rem;color:{C["orange"]};font-weight:700;'
                f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px">'
                f'Team-side flips ({len(team_paths)})</div>',
                unsafe_allow_html=True,
            )
            causal_icon = {"CAUSAL": "✓", "MIXED": "~",
                            "CORRELATIONAL": "⚠", "NONE": "—"}
            for p in team_paths:
                label = p.get("lever_label") or "?"
                causal = (p.get("causal_status") or "MIXED").upper()
                icon = causal_icon.get(causal, "?")
                magnitude = p.get("magnitude", 0) or 0
                base_p = p.get("baseline_prob", 0) or 0
                cf_p = p.get("counterfactual_prob", 0) or 0
                delta_pp = p.get("delta_prob_pp", 0) or 0
                cost = p.get("est_program_cost_usd")
                cost_str = f"${cost:,}" if cost else "—"
                lead = p.get("lead_time_days")
                lead_str = f"{lead}d" if lead is not None else "—"
                projected_risk = (p.get("projected_risk_level") or "?")
                projected_verdict = p.get("projected_verdict") or "?"
                projected_rule = p.get("projected_rule") or "—"
                feas = (p.get("feasibility") or "medium").lower()
                feas_c = feas_color.get(feas, C["text2"])
                vstyle = _verdict_badge_style(projected_verdict)
                st.markdown(
                    f'<div class="mrow" style="border-left:3px solid {feas_c}">'
                    f'<div style="flex:1">'
                    f'<div class="mrow-name">{label} '
                    f'<span style="font-size:0.7rem;background:#F0F0F5;padding:2px 8px;'
                    f'border-radius:8px;font-weight:600;color:#1D1D1F">{icon} {causal}</span> '
                    f'<span style="color:#86868B;font-size:0.78rem;'
                    f'font-weight:500">·&nbsp;mag {magnitude:.2f}</span></div>'
                    f'<div class="mrow-tier">'
                    f'risk drops <strong>{base_p:.0%}</strong> → '
                    f'<strong style="color:{C["green_dark"]}">{cf_p:.0%}</strong> '
                    f'(<strong style="color:{C["green_dark"]}">−{delta_pp:.1f}pp</strong>) '
                    f'→ <strong>{projected_risk}</strong>'
                    f'</div>'
                    f'<div style="font-size:0.78rem;color:#6E6E73;margin-top:4px">'
                    f'<span class="memo-verdict-badge" style="{vstyle};font-size:0.68rem">{projected_verdict}</span> '
                    f'<code style="font-size:0.7rem;background:#F0F0F5;padding:1px 6px;'
                    f'border-radius:6px;color:#1D1D1F">{projected_rule}</code> '
                    f'· {cost_str} · {lead_str} lead · feasibility '
                    f'<strong style="color:{feas_c}">{feas}</strong>'
                    f'</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Right: chart of flip impact, ordered by feasibility then size ----
    with col_r:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        rows = []
        for p in cand_paths:
            d = p.get("fit_delta_points") or 0
            rows.append({
                "lever": (p.get("lever") or "?")[:32],
                "approach": "Candidate",
                "magnitude": float(d),
                "unit": "pts fit",
                "feasibility": (p.get("feasibility") or "medium").lower(),
            })
        for p in team_paths:
            rows.append({
                "lever": (p.get("lever_label") or "?")[:32],
                "approach": "Team",
                "magnitude": float(p.get("delta_prob_pp") or 0),
                "unit": "pp risk",
                "feasibility": (p.get("feasibility") or "medium").lower(),
            })
        # Sort by feasibility (high first), then by absolute magnitude desc.
        rows.sort(key=lambda r: (feas_rank.get(r["feasibility"], 3),
                                   -abs(r["magnitude"])))
        if rows:
            df = pd.DataFrame(rows)
            colors = [feas_color.get(f, C["text2"])
                      for f in df["feasibility"][::-1]]
            labels = [f"[{a[0]}] {l}"
                      for a, l in zip(df["approach"][::-1], df["lever"][::-1])]
            mags = list(df["magnitude"][::-1])
            units = list(df["unit"][::-1])
            text_labels = [f"{m:.1f} {u}" for m, u in zip(mags, units)]
            fig = go.Figure(go.Bar(
                y=labels, x=mags, orientation="h",
                marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
                text=text_labels, textposition="outside",
                customdata=[[u, f] for u, f in zip(units,
                                                     list(df["feasibility"][::-1]))],
                hovertemplate="<b>%{y}</b><br>Δ %{x:.1f} %{customdata[0]}"
                              "<br>Feasibility: %{customdata[1]}<extra></extra>",
            ))
            fig.update_layout(
                title="Flip impact by feasibility",
                xaxis=dict(title="Δ (fit pts or risk pp)"),
                yaxis=dict(title=""),
            )
            style_fig(fig, max(280, 60 + 38 * len(rows)))
            st.plotly_chart(fig, width="stretch")
        else:
            st.markdown(
                '<div style="color:#86868B;font-size:0.85rem;padding:20px 0">'
                'No flip levers available.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def _render_joint_hire_memo(memo, show_trace=True):
    """Joint-hire renderer.

    Sections A/B/C keep the existing unified-card HTML rendering for
    visual continuity, while section D (Counterfactual flip explainer)
    is upgraded to chart+cards when it carries structured paths.
    """
    verdict_html = ""
    if memo.verdict:
        vstyle = _verdict_badge_style(memo.verdict)
        verdict_html = (
            f'<div class="memo-verdict-row">'
            f'<span class="memo-verdict-badge" style="{vstyle}">{memo.verdict}</span>'
            + (f'<span class="memo-rule-badge">{memo.rule_applied}</span>'
               if memo.rule_applied else "")
            + '</div>'
        )

    # Identify a counterfactual section by its structured paths metric.
    # Anything else stays inside the unified card.
    cf_section = None
    main_sections = []
    for s in memo.sections:
        m = s.metrics or {}
        if "candidate_paths" in m or "team_paths" in m:
            cf_section = s
        else:
            main_sections.append(s)

    sections_html = ""
    section_classes = ["memo-section-a", "memo-section-b", "memo-section-c"]
    for i, s in enumerate(main_sections):
        body = s.narrative.replace("\n", "<br>")
        cls = section_classes[i] if i < len(section_classes) else ""
        sections_html += (
            f'<div class="memo-section {cls}">'
            f'<div class="memo-section-title">{s.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div>'
        )

    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )

    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        + verdict_html
        + sections_html
        + '<div class="memo-footer">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    # Counterfactual section rendered as its own chart+cards block — only
    # appears for DEFER/DECLINE memos where the brain produced flip paths.
    if cf_section is not None:
        _render_counterfactual_block(cf_section)

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def _render_agent_memo(memo, show_trace=True):
    """Render an AgentMemo as a rich HTML card with sections + footer."""
    # Triage / shortlist memos carry a structured rankings list — render
    # them as a chart + styled cards instead of the raw markdown table.
    if memo.memo_type in ("triage", "shortlist"):
        _render_ranked_memo(memo, show_trace=show_trace)
        return
    # Retention plan memo — hero KPIs + chart + cards for both tables.
    if memo.memo_type == "retention_plan":
        _render_retention_plan_memo(memo, show_trace=show_trace)
        return
    # Risk-scan memo — KPIs + chart + ranking cards + deep-dive cards + actions.
    if memo.memo_type == "risk_scan":
        _render_risk_scan_memo(memo, show_trace=show_trace)
        return
    # Multi-brain consensus memo — 3 side-by-side brain cards + consensus banner.
    if memo.memo_type == "consensus_decision":
        _render_consensus_decision_memo(memo, show_trace=show_trace)
        return
    # Joint hire memo — keep sections A/B/C in unified card, but render
    # section D (counterfactual) as chart+cards when paths are structured.
    if memo.memo_type == "hire_decision":
        _render_joint_hire_memo(memo, show_trace=show_trace)
        return

    verdict_html = ""
    if memo.verdict:
        vstyle = _verdict_badge_style(memo.verdict)
        verdict_html = (
            f'<div class="memo-verdict-row">'
            f'<span class="memo-verdict-badge" style="{vstyle}">{memo.verdict}</span>'
            + (f'<span class="memo-rule-badge">{memo.rule_applied}</span>'
               if memo.rule_applied else "")
            + '</div>'
        )

    sections_html = ""
    section_classes = ["memo-section-a", "memo-section-b", "memo-section-c"]
    for i, s in enumerate(memo.sections):
        body = s.narrative.replace("\n", "<br>")
        cls = section_classes[i] if i < len(section_classes) else ""
        sections_html += (
            f'<div class="memo-section {cls}">'
            f'<div class="memo-section-title">{s.title}</div>'
            f'<div class="memo-section-body">{body}</div>'
            f'</div>'
        )

    conf = memo.confidence or {}
    footer_items = [
        f'<span class="memo-footer-item"><strong>Engine:</strong> Proprietary Brain v{memo.brain_version}</span>',
        f'<span class="memo-footer-item"><strong>Latency:</strong> {memo.total_elapsed_ms:.0f} ms</span>',
        f'<span class="memo-footer-item"><strong>Tool calls:</strong> {len(memo.execution_trace)}</span>',
        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 0</span>',
    ]
    if conf.get("level"):
        footer_items.append(
            f'<span class="memo-footer-item"><strong>Confidence:</strong> '
            f'{conf["level"]} ({conf.get("score", 0):.2f})</span>'
        )

    st.markdown(
        f'<div class="agent-memo">'
        f'<div class="memo-headline">{memo.headline}</div>'
        + verdict_html
        + sections_html
        + '<div class="memo-footer">'
        + "".join(footer_items)
        + '</div></div>',
        unsafe_allow_html=True,
    )

    if show_trace and memo.execution_trace:
        with st.expander(f"Execution trace ({len(memo.execution_trace)} tool calls)"):
            for step in memo.execution_trace:
                status = "✗" if step.had_error else "✓"
                color = "#D70015" if step.had_error else "#248A3D"
                st.markdown(
                    f'<div class="trace-row">'
                    f'<span class="trace-tool" style="color:{color}">{status} {step.tool}</span>'
                    f'<span class="trace-ms">{step.elapsed_ms:.1f} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if step.had_error and step.error_message:
                    st.markdown(
                        f'<div style="padding:4px 12px;font-size:0.72rem;color:#D70015">'
                        f'&nbsp;&nbsp;&nbsp;↳ {step.error_message}</div>',
                        unsafe_allow_html=True,
                    )


def render_boardroom_brief_page(employees, monthly, individual):
    """The Proprietary Agent Brain interface — LLM-free, deterministic
    boardroom memos. Five scenarios match `main.py --mode agent`:

      - Joint Hire Analysis   (Agent 1 + Agent 2 → hire / retention / backfill)
      - Workforce Risk Scan   (cross-dept prioritized action plan)
      - Quarterly Retention   ($-bounded intervention portfolio)
      - Candidate Shortlist   (rank N applicants for one req)
      - Role Triage           (rank N roles for one candidate)
    """
    import os as _os

    st.markdown('<h1><span class="gradient-title">Boardroom Brief</span></h1>',
                unsafe_allow_html=True)
    st.markdown(
        "Deterministic agent memos generated by the Proprietary Brain — "
        "LLM-free orchestration over 23 tools across 6 scenarios. Every "
        "verdict traces to a named rule in the 20-cell decision matrix. "
        "Joint-hire decisions can be cross-checked via Multi-Brain "
        "Consensus (three independent decision procedures); Workforce "
        "Risk Scan adds multi-policy threshold sensitivity to surface "
        "borderline departments for human review."
    )
    st.markdown("")

    # Scenario picker first; engine is shown only when alternatives exist
    # for that scenario. Currently only Joint Hire Analysis has a CrewAI
    # implementation, so the engine selector hides for every other scenario.
    has_claude = bool(_os.getenv("ANTHROPIC_API_KEY"))
    CREWAI_SUPPORTED_SCENARIOS = {"Joint Hire Analysis"}

    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        scenario = st.selectbox(
            "Scenario",
            [
                "Joint Hire Analysis",
                "Workforce Risk Scan",
                "Quarterly Retention Plan",
                "Candidate Shortlist",
                "Role Triage",
                "Multi-Brain Consensus",
            ],
        )
    crewai_available = has_claude and scenario in CREWAI_SUPPORTED_SCENARIOS
    with c2:
        if crewai_available:
            engine = st.selectbox(
                "Engine",
                ["Proprietary Brain (recommended)",
                 "CrewAI + Claude Sonnet 4"],
                help=(
                    "Proprietary Brain: deterministic, 1-3 s per memo "
                    "(measured), zero external API calls.\n\n"
                    "CrewAI + Claude: LLM reasoning, ~15-30 s, requires "
                    "ANTHROPIC_API_KEY (Joint Hire Analysis only)."
                ),
            )
        else:
            engine = "Proprietary Brain (recommended)"
            note = (
                "Proprietary Brain only — CrewAI is wired for "
                f"{', '.join(sorted(CREWAI_SUPPORTED_SCENARIOS))} only."
                if not has_claude
                else (
                    "Proprietary Brain only — this scenario has no CrewAI "
                    "implementation."
                )
            )
            st.markdown(
                '<div style="font-size:0.78rem;color:#86868B;'
                'text-transform:uppercase;letter-spacing:0.06em;'
                'font-weight:500;margin-bottom:4px">Engine</div>'
                '<div style="background:#FFFFFF;border:1px solid #D2D2D7;'
                'border-radius:10px;padding:10px 14px;color:#1D1D1F;'
                'font-size:0.92rem">Proprietary Brain (recommended)</div>'
                f'<div style="font-size:0.72rem;color:#86868B;'
                f'margin-top:6px;line-height:1.4">{note}</div>',
                unsafe_allow_html=True,
            )

    # Engine status pill
    is_proprietary = engine.startswith("Proprietary")
    engine_label = "PROPRIETARY BRAIN — 0 LLM CALLS" if is_proprietary else "CREWAI + CLAUDE SONNET 4"
    cls = "engine-pill-active" if is_proprietary else "engine-pill-inactive"
    st.markdown(
        f'<div style="margin:8px 0 18px 0">'
        f'<span class="{cls}">{engine_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Load models + register tool backends
    forecast = load_forecast()
    ner = load_ner()
    sbert = load_sbert()
    if forecast is None or ner is None or sbert is None:
        st.error("DL models not loaded. Run `python main.py --mode train` first.")
        return

    from tools.talent_tools import set_ner_engine, set_sbert_matcher
    from tools.forecast_tools import (
        set_forecast_engine, set_workforce_data,
    )
    set_ner_engine(ner)
    set_sbert_matcher(sbert)
    set_forecast_engine(forecast)
    set_workforce_data(monthly, individual, employees)

    from data.sample_resumes import get_active_resumes, get_active_jds
    SAMPLE_RESUMES = get_active_resumes()
    SAMPLE_JOB_DESCRIPTIONS = get_active_jds()

    # Scenario-specific input widgets
    if scenario == "Joint Hire Analysis":
        i1, i2, i3 = st.columns(3, gap="medium")
        with i1:
            cand_name = st.selectbox(
                "Candidate",
                [r["name"] for r in SAMPLE_RESUMES],
                key="bbr_c",
            )
        with i2:
            job_title = st.selectbox(
                "Position",
                [j["title"] for j in SAMPLE_JOB_DESCRIPTIONS],
                key="bbr_j",
            )
        with i3:
            dept = st.selectbox(
                "Receiving Department",
                sorted(monthly["department"].unique()),
                key="bbr_d",
            )
        cand = next(r for r in SAMPLE_RESUMES if r["name"] == cand_name)
        job = next(j for j in SAMPLE_JOB_DESCRIPTIONS if j["title"] == job_title)
    elif scenario == "Workforce Risk Scan":
        dept_filter = st.multiselect(
            "Limit to departments (empty = all)",
            sorted(monthly["department"].unique()),
            key="bbr_scan_d",
        )
    elif scenario == "Quarterly Retention Plan":
        budget = st.number_input(
            "Budget ($)",
            min_value=10_000, max_value=10_000_000,
            value=500_000, step=50_000, key="bbr_budget",
        )
        dept_filter = st.multiselect(
            "Focus departments (empty = all)",
            sorted(monthly["department"].unique()),
            key="bbr_plan_d",
        )
    elif scenario == "Candidate Shortlist":
        job_title = st.selectbox(
            "Open Position",
            [j["title"] for j in SAMPLE_JOB_DESCRIPTIONS],
            key="bbr_short_j",
        )
        job = next(j for j in SAMPLE_JOB_DESCRIPTIONS if j["title"] == job_title)
    elif scenario == "Role Triage":
        cand_name = st.selectbox(
            "Candidate",
            [r["name"] for r in SAMPLE_RESUMES],
            key="bbr_tri_c",
        )
        cand = next(r for r in SAMPLE_RESUMES if r["name"] == cand_name)
    elif scenario == "Multi-Brain Consensus":
        st.markdown(
            '<div style="color:#86868B;font-size:0.78rem;line-height:1.5;'
            'margin-bottom:6px">Runs three independent decision procedures '
            '(Sophisticated, Conservative, Heuristic baseline) on the same '
            'hire and flags whether they converge. EU AI Act Art. 14 '
            'human-oversight pattern.</div>',
            unsafe_allow_html=True,
        )
        i1, i2, i3 = st.columns(3, gap="medium")
        with i1:
            cand_name = st.selectbox(
                "Candidate",
                [r["name"] for r in SAMPLE_RESUMES],
                key="bbr_mb_c",
            )
        with i2:
            job_title = st.selectbox(
                "Position",
                [j["title"] for j in SAMPLE_JOB_DESCRIPTIONS],
                key="bbr_mb_j",
            )
        with i3:
            dept = st.selectbox(
                "Receiving Department",
                sorted(monthly["department"].unique()),
                key="bbr_mb_d",
            )
        cand = next(r for r in SAMPLE_RESUMES if r["name"] == cand_name)
        job = next(j for j in SAMPLE_JOB_DESCRIPTIONS if j["title"] == job_title)

    run = st.button("Generate Memo", key="bbr_run")
    if not run:
        st.markdown(
            f'''<div class="section-card" style="min-height:220px;display:flex;
            flex-direction:column;align-items:center;justify-content:center;gap:14px">
                <div style="width:48px;height:48px;border-radius:14px;
                background:linear-gradient(135deg,#EBF5FF,#F5EEFA);display:flex;
                align-items:center;justify-content:center;font-size:1.5rem">📋</div>
                <div style="text-align:center;color:#86868B;font-size:0.85rem;
                line-height:1.6">
                Click <b>Generate Memo</b> to run the <b>{engine}</b><br>
                on scenario: <b>{scenario}</b></div>
            </div>''',
            unsafe_allow_html=True,
        )
        return

    # Route to the engine
    if is_proprietary:
        from brain import build_brain
        enable_learning = st.session_state.get("learning_enabled", True)
        brain = build_brain(verbose=False, enable_learning=enable_learning)

        with st.spinner(f"Running Proprietary Brain — {scenario}..."):
            if scenario == "Joint Hire Analysis":
                memo = brain.joint_hire_analysis(
                    resume_text=cand["text"], job_text=job["text"],
                    department=dept,
                )
            elif scenario == "Workforce Risk Scan":
                memo = brain.workforce_risk_scan(
                    departments=dept_filter if dept_filter else None,
                )
            elif scenario == "Quarterly Retention Plan":
                memo = brain.quarterly_retention_plan(
                    budget_usd=int(budget),
                    focus_depts=dept_filter if dept_filter else None,
                )
            elif scenario == "Candidate Shortlist":
                cands = [
                    {"id": str(i), "name": r["name"], "text": r["text"]}
                    for i, r in enumerate(SAMPLE_RESUMES)
                ]
                memo = brain.rank_candidates_for_req(
                    job_text=job["text"], candidates=cands,
                )
            elif scenario == "Role Triage":
                jobs = [
                    {"id": str(i), "title": j["title"], "text": j["text"]}
                    for i, j in enumerate(SAMPLE_JOB_DESCRIPTIONS)
                ]
                memo = brain.match_candidate_across_reqs(
                    resume_text=cand["text"], jobs=jobs,
                )
            elif scenario == "Multi-Brain Consensus":
                memo = brain.multi_brain_consensus(
                    resume_text=cand["text"], job_text=job["text"],
                    department=dept,
                )
        # Stash for the HITL page "Submit Feedback" tab.
        st.session_state["last_memo"] = memo
        # Also persist a summary to the memo-history store so the HITL
        # Review Queue can list it across sessions.
        try:
            from feedback import MemoHistoryStore
            from datetime import datetime, timezone
            hist = MemoHistoryStore()
            hist.log({
                "memo_id": (memo.metrics or {}).get("memo_id"),
                "timestamp": datetime.now(timezone.utc).replace(
                    microsecond=0).isoformat().replace("+00:00", "Z"),
                "memo_type": memo.memo_type,
                "headline": memo.headline,
                "verdict": memo.verdict,
                "rule_applied": memo.rule_applied,
                "department": (memo.metrics or {}).get("department"),
                "review_priority": (memo.metrics or {}).get("review_priority"),
                "learning_version": (memo.metrics or {}).get("learning_version"),
                "feedback_state_hash": (memo.metrics or {}).get("feedback_state_hash"),
                "n_adjustments_applied": len(
                    (memo.metrics or {}).get("adjustments_applied") or []
                ),
            })
        except Exception:
            pass
        _render_agent_memo(memo)
    else:
        # CrewAI + Claude path — delegates to agents.py
        st.info(
            "CrewAI + Claude path runs the full ReAct loop with live LLM calls. "
            "This can take 15-30 seconds for a joint-hire scenario. Using the "
            "existing demo wiring from `agents.py`."
        )
        with st.spinner("Running CrewAI + Claude Sonnet 4..."):
            import time as _time
            t0 = _time.perf_counter()
            try:
                from crewai import LLM as _LLM
                from agents import (
                    create_talent_agent, create_forecast_agent,
                    create_combined_task, build_crew,
                )
                llm = _LLM(
                    model="anthropic/claude-sonnet-4-20250514",
                    api_key=_os.getenv("ANTHROPIC_API_KEY"),
                    temperature=0.3,
                )
                if scenario == "Joint Hire Analysis":
                    talent_agent = create_talent_agent(llm=llm)
                    forecast_agent = create_forecast_agent(llm=llm)
                    t_task, f_task = create_combined_task(
                        talent_agent, forecast_agent,
                        cand["text"], job["text"], dept,
                    )
                    crew = build_crew(talent_agent, forecast_agent,
                                      [t_task, f_task], verbose=False)
                    result = crew.kickoff()
                    elapsed = (_time.perf_counter() - t0) * 1000
                    st.markdown('<div class="agent-memo">', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="memo-headline">CrewAI + Claude — Joint Hire Memo</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="memo-section memo-section-a">'
                        f'<div class="memo-section-title">Unified Memo</div>'
                        f'<div class="memo-section-body">{str(result)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="memo-footer">'
                        f'<span class="memo-footer-item"><strong>Engine:</strong> CrewAI + Claude Sonnet 4</span>'
                        f'<span class="memo-footer-item"><strong>Latency:</strong> {elapsed:.0f} ms</span>'
                        f'<span class="memo-footer-item"><strong>LLM calls:</strong> 1+</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        f"CrewAI path for scenario '{scenario}' is not wired. "
                        "Use the Proprietary Brain for this scenario."
                    )
            except Exception as e:
                st.error(f"CrewAI path failed: {e}")


# ─── HUMAN-IN-THE-LEAD PAGE ─────────────────────────────────────────
def _priority_badge(priority: str) -> str:
    """Color-coded badge HTML for a review priority."""
    m = {
        "HIGH":   ("#D70015", "🔴 HIGH"),
        "NORMAL": ("#FF9500", "🟡 NORMAL"),
        "LOW":    ("#248A3D", "🟢 LOW"),
    }
    bg, label = m.get((priority or "").upper(), ("#86868B", priority or "—"))
    return (f'<span style="background:{bg};color:#fff;padding:2px 10px;'
            f'border-radius:8px;font-size:0.75rem;font-weight:600">'
            f'{label}</span>')


def _engine_toggle_sidebar() -> bool:
    """Reusable sidebar toggle for the learning engine. Returns True when
    adaptive mode is selected. Stored in session state so all pages see
    the same setting."""
    if "learning_enabled" not in st.session_state:
        st.session_state["learning_enabled"] = True
    st.session_state["learning_enabled"] = st.sidebar.toggle(
        "Learning engine",
        value=st.session_state["learning_enabled"],
        help=(
            "ON (adaptive): Human-in-the-Lead feedback active. Brain "
            "decisions may be adjusted by prior human corrections.\n\n"
            "OFF (static): byte-identical to the pre-feedback brain. "
            "Use for reproducibility snapshots and regulatory audits."
        ),
    )
    state_label = "🟢 ADAPTIVE" if st.session_state["learning_enabled"] else "⚫ STATIC"
    st.sidebar.markdown(
        f'<div style="font-size:0.72rem;color:#6E6E73;padding:4px 0">'
        f'Engine state: <strong>{state_label}</strong></div>',
        unsafe_allow_html=True,
    )
    return st.session_state["learning_enabled"]


def _build_learning_visuals(events, adjustments, memo_history_rows,
                              auto_apply_threshold):
    """Build five Plotly figures visualizing the brain's continual-
    learning progress:

      1. Events over time — cumulative stacked area by event type
      2. Decision-matrix heatmap — vote counts per (fit_tier × risk_level)
         with applied cells highlighted green
      3. Pending-corrections progress bars — top pending cells with
         vote count vs threshold
      4. Review-queue priority donut — memo review-priority mix
      5. Learning version timeline — milestones as the brain evolves

    Returns a dict of {name: plotly.graph_objects.Figure}. Each figure
    degrades gracefully when the relevant slice of data is empty.
    """
    from collections import Counter
    figs: Dict[str, "go.Figure"] = {}

    # ---- 1. Events over time ────────────────────────────────────────
    fig_events = go.Figure()
    if events:
        df = pd.DataFrame([
            {"ts": e.timestamp, "type": e.event_type} for e in events
        ])
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        if not df.empty:
            span_h = ((df["ts"].max() - df["ts"].min()).total_seconds()
                       / 3600.0)
            freq = "h" if span_h < 48 else "D"
            df["bucket"] = df["ts"].dt.floor(freq)
            pivot = (df.groupby(["bucket", "type"]).size()
                     .unstack(fill_value=0).sort_index().cumsum())
            palette = {
                "verdict_correction":     "#0071E3",
                "rule_override":          "#AF52DE",
                "confidence_calibration": "#5AC8FA",
                "causal_update":          "#34C759",
                "new_intervention":       "#FF9500",
                "general_comment":        "#86868B",
            }
            for col in pivot.columns:
                fig_events.add_trace(go.Scatter(
                    x=pivot.index, y=pivot[col], mode="lines",
                    name=col.replace("_", " "),
                    stackgroup="one",
                    line=dict(width=0.6, color=palette.get(col, "#86868B")),
                    fillcolor=palette.get(col, "#86868B"),
                    opacity=0.85,
                    hovertemplate="%{x|%b %d, %H:%M}<br>"
                                  + col + ": %{y}<extra></extra>",
                ))
    else:
        fig_events.add_annotation(
            text="No feedback events logged yet — submit one to see growth.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#86868B", size=12),
        )
    fig_events.update_layout(title="Cumulative feedback activity",
                              yaxis_title="events")
    figs["events_timeline"] = style_fig(fig_events, h=280)

    # ---- 2. Decision-matrix heatmap ─────────────────────────────────
    fit_tiers = ["STRONG_HIRE", "HIRE", "INTERVIEW",
                  "CONDITIONAL", "DO_NOT_ADVANCE"]
    risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # Aggregate votes per cell from applied verdict_correction events
    cell_votes: Dict[tuple, Counter] = {}
    for e in events:
        if e.event_type != "verdict_correction" or not e.applied_at:
            continue
        ov = e.original_value or {}
        cv = e.corrected_value or {}
        if not (isinstance(ov, dict) and isinstance(cv, dict)):
            continue
        ft, rl = ov.get("fit_tier"), ov.get("risk_level")
        nv = cv.get("verdict")
        if ft and rl and nv:
            cell_votes.setdefault((ft, rl), Counter())[nv] += 1

    rule_overrides = adjustments.get("rule_overrides", {})
    z: List[List[float]] = []
    labels: List[List[str]] = []
    for ft in fit_tiers:
        row_z, row_lbl = [], []
        for rl in risk_levels:
            key = f"{ft}|{rl}"
            if key in rule_overrides:
                # Applied cell — saturate colour.
                row_z.append(auto_apply_threshold + 0.5)
                new_v = rule_overrides[key]["new_verdict"]
                row_lbl.append(f"✓ APPLIED<br>→ {new_v}")
                continue
            ctr = cell_votes.get((ft, rl))
            if not ctr:
                row_z.append(0)
                row_lbl.append("")
                continue
            top_v, votes = ctr.most_common(1)[0]
            row_z.append(min(votes, auto_apply_threshold))
            row_lbl.append(f"{votes}/{auto_apply_threshold}<br>→ {top_v}")
        z.append(row_z)
        labels.append(row_lbl)

    fig_matrix = go.Figure(data=go.Heatmap(
        z=z, x=risk_levels, y=fit_tiers,
        text=labels, texttemplate="%{text}",
        textfont=dict(size=10, color="#1D1D1F"),
        colorscale=[
            [0.0, "#F5F5F7"],
            [0.25, "#EBF5FF"],
            [0.50, "#B3DAFF"],
            [0.75, "#0071E3"],
            [1.0, "#34C759"],
        ],
        zmin=0, zmax=auto_apply_threshold + 0.5,
        showscale=False,
        hovertemplate="Fit tier: %{y}<br>Risk level: %{x}<br>"
                       "%{text}<extra></extra>",
    ))
    fig_matrix.update_layout(
        title=f"Decision matrix — votes toward auto-apply (threshold {auto_apply_threshold})",
        xaxis=dict(side="top", title=None),
        yaxis=dict(autorange="reversed", title=None),
    )
    figs["decision_matrix"] = style_fig(fig_matrix, h=280)

    # ---- 3. Pending-corrections progress bars ───────────────────────
    pending_rows = []
    for (ft, rl), ctr in cell_votes.items():
        key = f"{ft}|{rl}"
        if key in rule_overrides:
            continue  # already applied — shown in matrix
        top_v, votes = ctr.most_common(1)[0]
        pending_rows.append({
            "label": f"{ft} × {rl} → {top_v}",
            "votes": votes,
        })
    pending_rows.sort(key=lambda r: -r["votes"])
    pending_rows = pending_rows[:8]

    fig_progress = go.Figure()
    if pending_rows:
        bar_colors = [
            "#FF9500" if r["votes"] >= max(1, auto_apply_threshold - 1)
            else "#0071E3"
            for r in pending_rows
        ]
        fig_progress.add_trace(go.Bar(
            y=[r["label"] for r in pending_rows],
            x=[r["votes"] for r in pending_rows],
            orientation="h",
            marker=dict(color=bar_colors,
                        line=dict(color="#D2D2D7", width=0.5)),
            text=[f"{r['votes']}/{auto_apply_threshold}"
                  for r in pending_rows],
            textposition="outside",
            hovertemplate="%{y}<br>%{x} vote(s)<extra></extra>",
        ))
        fig_progress.add_vline(
            x=auto_apply_threshold, line_dash="dash",
            line_color="#248A3D", line_width=2,
            annotation_text="auto-apply",
            annotation_position="top right",
            annotation=dict(font=dict(color="#248A3D", size=11)),
        )
    else:
        fig_progress.add_annotation(
            text="No pending corrections — all votes have been applied "
                 "or none logged yet.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#86868B", size=12),
        )
    fig_progress.update_layout(
        title="Pending corrections — votes to threshold",
        xaxis=dict(range=[0, auto_apply_threshold + 0.8],
                    title="votes"),
        yaxis=dict(autorange="reversed", title=None),
    )
    figs["progress_bars"] = style_fig(fig_progress, h=280)

    # ---- 4. Review-queue priority donut ─────────────────────────────
    prio_counter = Counter()
    for r in memo_history_rows or []:
        prio_counter[r.get("review_priority") or "—"] += 1
    fig_prio = go.Figure()
    if sum(prio_counter.values()) > 0:
        order = ["HIGH", "NORMAL", "LOW", "—"]
        labels_ordered = [p for p in order if p in prio_counter]
        values = [prio_counter[p] for p in labels_ordered]
        colors = {"HIGH": "#D70015", "NORMAL": "#FF9500",
                   "LOW": "#248A3D", "—": "#86868B"}
        fig_prio.add_trace(go.Pie(
            labels=labels_ordered, values=values,
            marker=dict(colors=[colors[p] for p in labels_ordered],
                         line=dict(color="#FFFFFF", width=2)),
            hole=0.55,
            textinfo="label+value",
            textfont=dict(size=12, color="#1D1D1F"),
            hovertemplate="%{label}: %{value} memo(s)<extra></extra>",
        ))
    else:
        fig_prio.add_annotation(
            text="No memos in history yet — run a scenario first.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#86868B", size=12),
        )
    fig_prio.update_layout(title="Review queue priority mix",
                            showlegend=False)
    figs["priority_donut"] = style_fig(fig_prio, h=280)

    # ---- 5. Learning version timeline ───────────────────────────────
    # One dot per applied event, labelled with the resulting version.
    # Separate milestone markers when rule_overrides fire.
    fig_version = go.Figure()
    if events:
        applied_events = [e for e in events if e.applied_at]
        if applied_events:
            df_v = pd.DataFrame([{
                "ts": e.applied_at,
                "type": e.event_type,
                "version": f"v{i + 1}",
            } for i, e in enumerate(applied_events)])
            df_v["ts"] = pd.to_datetime(df_v["ts"], errors="coerce")
            df_v = df_v.dropna(subset=["ts"])
            # Base scatter — every applied event
            fig_version.add_trace(go.Scatter(
                x=df_v["ts"], y=df_v["version"],
                mode="markers+text",
                marker=dict(size=12, color="#0071E3", symbol="circle",
                            line=dict(color="#FFFFFF", width=1)),
                text=df_v["type"].str.replace("_", " "),
                textposition="middle right",
                textfont=dict(size=10, color="#6E6E73"),
                name="Applied event",
                hovertemplate="%{text}<br>Version: %{y}<br>"
                              "Applied: %{x|%b %d, %H:%M}<extra></extra>",
            ))
            # Milestone markers for every rule override currently active
            # (shown as gold diamonds at their latest supporting event)
            override_event_ids = set()
            for ov in rule_overrides.values():
                override_event_ids.update(ov.get("event_ids", []))
            milestones = df_v[
                df_v["ts"].index.map(
                    lambda i: applied_events[i].event_id in override_event_ids
                )
            ] if override_event_ids else pd.DataFrame()
            if not milestones.empty:
                # Keep only the latest per event-id cluster
                fig_version.add_trace(go.Scatter(
                    x=milestones["ts"], y=milestones["version"],
                    mode="markers",
                    marker=dict(size=18, color="#34C759",
                                symbol="diamond",
                                line=dict(color="#FFFFFF", width=2)),
                    name="Rule override fired",
                    hovertemplate="Rule override milestone<br>"
                                   "Version: %{y}<extra></extra>",
                ))
    else:
        fig_version.add_annotation(
            text="No applied events — version will be v0-empty until "
                 "the first feedback arrives.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#86868B", size=12),
        )
    fig_version.update_layout(
        title="Brain learning timeline (applied events → version bumps)",
        yaxis=dict(title="version", tickmode="linear"),
        xaxis=dict(title="time"),
    )
    figs["version_timeline"] = style_fig(fig_version, h=280)

    return figs


def render_human_in_the_lead_page(employees, monthly, individual):
    """
    Human-in-the-Lead control surface for the continual-learning system.
    Three tabs:
      1. Submit Feedback — review a memo + submit a correction
      2. Review Queue — HIGH_PRIORITY items the brain wants humans to see
      3. Learning State — current adjustments, votes, rollback

    This page satisfies EU AI Act Art. 14 (human oversight) by making every
    adjustment visible, every decision traceable to its source feedback
    events, and rollback available with one click.
    """
    st.markdown('<h1><span class="gradient-title">Human-in-the-Lead</span></h1>',
                unsafe_allow_html=True)
    st.markdown(
        "The brain learns from human corrections — transparently, reversibly, "
        "and only when evidence crosses the voting threshold. You are always "
        "in the lead."
    )
    st.markdown("")

    from feedback import (
        FeedbackStore, MemoHistoryStore, LearningEngine,
        capture_verdict_correction, capture_rule_override,
        capture_confidence_calibration, capture_causal_update,
        capture_new_intervention, capture_general_comment,
        rollback_last_adjustment, USER_ROLES, AUTO_APPLY_THRESHOLD,
        CAUSAL_STATUS_FLIP_N,
    )

    store = FeedbackStore()
    memo_history = MemoHistoryStore()
    engine = LearningEngine(store)
    adjustments = engine.compute_adjustments(force=True)

    # ── header banner: current learning state ─────────────────────────
    s = store.summarize()
    lv = adjustments["learning_version"]
    hash_short = adjustments["state_hash"][:6] if adjustments["state_hash"] else "—"
    n_overrides = len(adjustments["rule_overrides"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Learning version", lv)
    with c2:
        st.metric("State hash", hash_short)
    with c3:
        st.metric("Events logged", s["total_events"])
    with c4:
        st.metric("Active overrides", n_overrides)

    st.markdown("---")

    tab_submit, tab_queue, tab_state = st.tabs(
        ["Submit Feedback", "Review Queue", "Learning State"]
    )

    # ═════════════════════════════════════════════════════════════════
    # TAB 1 — Submit Feedback
    # ═════════════════════════════════════════════════════════════════
    with tab_submit:
        st.markdown("### Provide a correction")
        st.markdown(
            f'Below-threshold corrections accumulate until **≥ '
            f'{AUTO_APPLY_THRESHOLD} independent same-direction** votes '
            "trigger auto-apply. High-impact types (rule_override, "
            "new_intervention) require explicit approval from a CHRO."
        )

        last_memo = st.session_state.get("last_memo")
        if last_memo is not None:
            with st.expander("Memo being reviewed (from Boardroom Brief)",
                              expanded=False):
                _render_agent_memo(last_memo, show_trace=False)

        col_l, col_r = st.columns([3, 2], gap="large")
        with col_l:
            event_type = st.selectbox(
                "Feedback type",
                [
                    "verdict_correction",
                    "confidence_calibration",
                    "causal_update",
                    "new_intervention",
                    "rule_override",
                    "general_comment",
                ],
                help=(
                    "verdict_correction: per-case override, crowdsourced\n"
                    "rule_override: blanket policy change (requires approval)\n"
                    "confidence_calibration: was the confidence level right?\n"
                    "causal_update: does an intervention actually work?\n"
                    "new_intervention: add a lever not in the library\n"
                    "general_comment: free-text note, no learning effect"
                ),
            )
        with col_r:
            user_id = st.text_input("Your user ID", value="demo_user")
            role = st.selectbox("Your role", list(USER_ROLES), index=0)

        st.markdown("")
        # ---- per-type form bodies ---------------------------------
        if event_type == "verdict_correction":
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                memo_id = st.text_input(
                    "Memo ID",
                    value=(last_memo.metrics.get("memo_id")
                           if last_memo and last_memo.metrics else ""),
                )
            with c2:
                dept = st.selectbox(
                    "Department",
                    sorted(monthly["department"].unique()),
                )
            with c3:
                fit_tier = st.selectbox(
                    "Fit tier",
                    ["STRONG_HIRE", "HIRE", "INTERVIEW",
                     "CONDITIONAL", "DO_NOT_ADVANCE"],
                )
            c4, c5 = st.columns(2, gap="medium")
            with c4:
                risk_level = st.selectbox(
                    "Risk level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                )
            with c5:
                orig_verdict = st.text_input(
                    "Original verdict the brain gave",
                    value=(last_memo.verdict if last_memo else ""),
                )
            corrected = st.selectbox(
                "What the verdict SHOULD be",
                ["HIRE", "CONDITIONAL_HIRE", "HIRE_WITH_RETENTION_PLAN",
                 "ADVANCE_TO_INTERVIEW", "ADVANCE_WITH_CAVEATS",
                 "ADVANCE_WITH_SKILL_PLAN", "DEFER", "DECLINE",
                 "FAST_TRACK"],
            )
            rationale = st.text_area("Rationale (required)", height=100)
            if st.button("Submit verdict correction", type="primary"):
                if not rationale.strip():
                    st.error("Rationale is required.")
                else:
                    capture_verdict_correction(
                        store, memo_id=memo_id, department=dept,
                        fit_tier=fit_tier, risk_level=risk_level,
                        original_verdict=orig_verdict,
                        corrected_verdict=corrected,
                        user_id=user_id, user_role=role,
                        rationale=rationale,
                    )
                    st.success(f"Logged. Current votes for ({fit_tier}, "
                                f"{risk_level}) — threshold is "
                                f"{AUTO_APPLY_THRESHOLD}.")
                    st.rerun()

        elif event_type == "confidence_calibration":
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                memo_id = st.text_input(
                    "Memo ID",
                    value=(last_memo.metrics.get("memo_id")
                           if last_memo and last_memo.metrics else ""),
                )
            with c2:
                orig_level = st.selectbox(
                    "Original confidence level",
                    ["high", "medium", "low"],
                )
            was_correct = st.radio(
                "Was the verdict correct in practice?",
                ["Yes", "No"], horizontal=True,
            )
            rationale = st.text_area("Rationale", height=80)
            if st.button("Submit calibration", type="primary"):
                capture_confidence_calibration(
                    store, memo_id=memo_id, original_level=orig_level,
                    was_correct=(was_correct == "Yes"),
                    user_id=user_id, user_role=role, rationale=rationale,
                )
                st.success("Logged.")
                st.rerun()

        elif event_type == "causal_update":
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                interv = st.selectbox(
                    "Intervention",
                    ["comp_adjustment", "satisfaction_program",
                     "engagement_initiative", "retention_bonus",
                     "flex_work", "knowledge_capture"],
                )
            with c2:
                new_status = st.selectbox(
                    "New causal status",
                    ["CAUSAL", "MIXED", "CORRELATIONAL", "NONE"],
                )
            st.info(
                f"Flip requires ≥ {CAUSAL_STATUS_FLIP_N} independent "
                "reports before it takes effect."
            )
            rationale = st.text_area(
                "What did you observe?",
                placeholder="e.g., 'Ran the satisfaction program in Q2; "
                            "attrition dropped 4pp vs control group.'",
                height=100,
            )
            if st.button("Submit causal update", type="primary"):
                if not rationale.strip():
                    st.error("Rationale is required.")
                else:
                    capture_causal_update(
                        store, intervention=interv, new_status=new_status,
                        user_id=user_id, user_role=role, rationale=rationale,
                    )
                    st.success("Logged.")
                    st.rerun()

        elif event_type == "new_intervention":
            st.warning("This event type requires CHRO approval before it "
                        "takes effect. Logged as pending approval.")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                name = st.text_input("Intervention name (snake_case)",
                                      value="sabbatical_option")
                description = st.text_input("Description",
                                              value="Paid sabbatical after 7 years")
                feature = st.selectbox("Feature affected",
                                         ["comp_ratio", "satisfaction",
                                          "engagement", "None"])
            with c2:
                cost = st.number_input("Cost per head (USD)",
                                         min_value=100, value=5_000, step=500)
                lead = st.number_input("Lead time (days)",
                                         min_value=0, value=60, step=15)
                causal = st.selectbox("Causal status",
                                        ["CAUSAL", "MIXED",
                                         "CORRELATIONAL", "NONE"])
            rationale = st.text_area("Rationale", height=80)
            if st.button("Submit new intervention", type="primary"):
                if not rationale.strip():
                    st.error("Rationale is required.")
                else:
                    capture_new_intervention(
                        store, name=name,
                        spec={
                            "description": description,
                            "feature_affected": (feature if feature != "None" else None),
                            "cost_per_head_per_unit": int(cost),
                            "lead_time_days": int(lead),
                            "causal_status": causal,
                        },
                        user_id=user_id, user_role=role, rationale=rationale,
                    )
                    st.success("Logged (pending CHRO approval).")
                    st.rerun()

        elif event_type == "rule_override":
            st.warning("This event type requires approval before it takes "
                        "effect. Logged as pending approval.")
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                fit_tier = st.selectbox(
                    "Fit tier",
                    ["STRONG_HIRE", "HIRE", "INTERVIEW",
                     "CONDITIONAL", "DO_NOT_ADVANCE"],
                    key="ro_ft",
                )
            with c2:
                risk_level = st.selectbox(
                    "Risk level",
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    key="ro_rl",
                )
            with c3:
                new_verdict = st.selectbox(
                    "New verdict for this cell",
                    ["HIRE", "CONDITIONAL_HIRE",
                     "HIRE_WITH_RETENTION_PLAN",
                     "ADVANCE_TO_INTERVIEW",
                     "ADVANCE_WITH_CAVEATS",
                     "DEFER", "DECLINE", "FAST_TRACK"],
                    key="ro_nv",
                )
            rationale = st.text_area("Rationale", height=80,
                                        key="ro_rationale")
            if st.button("Submit rule override", type="primary"):
                if not rationale.strip():
                    st.error("Rationale is required.")
                else:
                    capture_rule_override(
                        store, fit_tier=fit_tier, risk_level=risk_level,
                        new_verdict=new_verdict, user_id=user_id,
                        user_role=role, rationale=rationale,
                    )
                    st.success("Logged (pending approval).")
                    st.rerun()

        elif event_type == "general_comment":
            memo_id = st.text_input("Memo ID (optional)")
            dept = st.selectbox(
                "Department (optional)",
                ["(none)"] + sorted(monthly["department"].unique()),
            )
            comment = st.text_area("Comment", height=120)
            if st.button("Submit comment", type="primary"):
                if not comment.strip():
                    st.error("Comment is required.")
                else:
                    capture_general_comment(
                        store,
                        memo_id=memo_id or None,
                        comment=comment,
                        user_id=user_id, user_role=role,
                        department=(dept if dept != "(none)" else None),
                    )
                    st.success("Logged.")
                    st.rerun()

    # ═════════════════════════════════════════════════════════════════
    # TAB 2 — Review Queue
    # ═════════════════════════════════════════════════════════════════
    with tab_queue:
        st.markdown("### Memos awaiting human review")
        st.markdown(
            "The brain flags memos where human feedback would improve it "
            "most: low-confidence predictions, wide conformal CIs, and "
            "multi-brain disagreements. Review these first."
        )
        rows = memo_history.load_recent(200)
        if not rows:
            st.info(
                "No memos logged yet. Generate a memo in the Boardroom "
                "Brief page, then return here."
            )
        else:
            priority_rank = {"HIGH": 0, "NORMAL": 1, "LOW": 2, None: 3}
            rows_sorted = sorted(
                rows,
                key=lambda r: (priority_rank.get(r.get("review_priority"), 3),
                                r.get("timestamp") or ""),
            )
            st.markdown("")
            for r in rows_sorted[:30]:
                badge = _priority_badge(r.get("review_priority") or "—")
                adjusted = r.get("n_adjustments_applied", 0)
                adj_note = (f'<span style="color:#AF52DE;font-size:0.72rem;'
                            f'padding-left:8px">· {adjusted} adjustment(s)</span>'
                            if adjusted else "")
                st.markdown(
                    f'<div class="section-card" style="padding:12px 16px;'
                    f'margin-bottom:8px;display:flex;justify-content:space-between;'
                    f'align-items:center;gap:12px">'
                    f'<div style="flex:1">'
                    f'<div style="font-weight:600;color:#1D1D1F;font-size:0.9rem">'
                    f'{r.get("headline", "(no headline)")}</div>'
                    f'<div style="font-size:0.72rem;color:#6E6E73;padding-top:2px">'
                    f'<code>{r.get("memo_id", "—")}</code> · '
                    f'{r.get("memo_type", "?")} · '
                    f'{r.get("department") or "—"} · '
                    f'{r.get("timestamp") or "—"}{adj_note}'
                    f'</div></div>'
                    f'<div>{badge}</div></div>',
                    unsafe_allow_html=True,
                )

    # ═════════════════════════════════════════════════════════════════
    # TAB 3 — Learning State
    # ═════════════════════════════════════════════════════════════════
    with tab_state:
        # ── Visualizations ─────────────────────────────────────────
        st.markdown("### Learning progress — at a glance")
        st.markdown(
            "Five views of the brain's continual-learning state. Each "
            "updates live as feedback arrives."
        )
        visuals = _build_learning_visuals(
            events=store.load_all(),
            adjustments=adjustments,
            memo_history_rows=memo_history.load_recent(200),
            auto_apply_threshold=AUTO_APPLY_THRESHOLD,
        )
        row1_l, row1_r = st.columns(2, gap="medium")
        with row1_l:
            st.plotly_chart(visuals["events_timeline"],
                             width="stretch",
                             key="hitl_events_timeline")
        with row1_r:
            st.plotly_chart(visuals["decision_matrix"],
                             width="stretch",
                             key="hitl_decision_matrix")
        row2_l, row2_r = st.columns(2, gap="medium")
        with row2_l:
            st.plotly_chart(visuals["progress_bars"],
                             width="stretch",
                             key="hitl_progress_bars")
        with row2_r:
            st.plotly_chart(visuals["priority_donut"],
                             width="stretch",
                             key="hitl_priority_donut")
        st.plotly_chart(visuals["version_timeline"],
                         width="stretch",
                         key="hitl_version_timeline")
        st.markdown("---")

        # ── Text summary ───────────────────────────────────────────
        st.markdown("### Current adjustments active")
        if n_overrides == 0 and not adjustments["confidence_multipliers"] \
                and not adjustments["causal_updates"] \
                and not adjustments["new_interventions"]:
            st.info(
                "No adjustments have met the voting threshold yet. The brain "
                "is currently running with its base policy."
            )
        else:
            if adjustments["rule_overrides"]:
                st.markdown("**Rule overrides (decision-matrix cells):**")
                for key, ov in adjustments["rule_overrides"].items():
                    ft, rl = key.split("|")
                    src_icon = "✓ approved" if ov["source"].startswith("rule_override") \
                                else f"↑ {ov.get('evidence_votes', '?')} votes"
                    st.markdown(
                        f'- `{ft}` × `{rl}` → **`{ov["new_verdict"]}`** '
                        f'[{src_icon}]'
                    )
            if adjustments["confidence_multipliers"]:
                st.markdown("**Confidence recalibration:**")
                for lvl, mult in adjustments["confidence_multipliers"].items():
                    st.markdown(f'- `{lvl}` confidence de-rated × {mult}')
            if adjustments["causal_updates"]:
                st.markdown("**Causal status updates:**")
                for interv, u in adjustments["causal_updates"].items():
                    st.markdown(
                        f'- `{interv}` → **{u["new_status"]}** '
                        f'({u["votes"]} reports)'
                    )
            if adjustments["new_interventions"]:
                st.markdown("**User-added interventions (approved):**")
                for name, spec in adjustments["new_interventions"].items():
                    st.markdown(
                        f'- `{name}` ({spec.get("description", "")})'
                    )

        st.markdown("")
        st.markdown("### Pending corrections (below threshold)")
        events = store.load_all()
        pending_verdict = [e for e in events
                            if e.event_type == "verdict_correction"]
        from collections import Counter
        vote_counter: dict = {}
        for e in pending_verdict:
            ov = e.original_value or {}
            cv = e.corrected_value or {}
            if isinstance(ov, dict) and isinstance(cv, dict):
                key = f'{ov.get("fit_tier")}|{ov.get("risk_level")}'
                vote_counter.setdefault(key, Counter())[cv.get("verdict")] += 1
        pending_lines = []
        for key, ctr in vote_counter.items():
            if key in adjustments["rule_overrides"]:
                continue  # already applied, shown above
            top_verdict, votes = ctr.most_common(1)[0]
            pending_lines.append(
                f'- `{key}` → `{top_verdict}` '
                f'({votes}/{AUTO_APPLY_THRESHOLD} votes)'
            )
        if pending_lines:
            st.markdown("\n".join(pending_lines))
        else:
            st.markdown("*(none)*")

        st.markdown("")
        st.markdown("### Governance")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            if st.button("Rollback last applied adjustment",
                         help="Unapply the most recent applied event. "
                              "The event stays in the log (audit trail); "
                              "only its applied_at flag is cleared."):
                rolled = rollback_last_adjustment(store)
                if rolled:
                    st.success(f"Rolled back event `{rolled.event_id}` "
                                f"({rolled.event_type}).")
                    st.rerun()
                else:
                    st.info("Nothing to roll back.")
        with c2:
            st.download_button(
                "Export feedback log (JSONL)",
                data=(store.path.read_bytes() if store.path.exists() else b""),
                file_name="feedback_log.jsonl",
                mime="application/jsonl",
                help="Full audit trail — immutable append-only record.",
            )

        # Pending approvals surface (rule_override + new_intervention)
        pending_approval = [e for e in events
                             if e.event_type in ("rule_override",
                                                  "new_intervention")
                             and not e.approved_at]
        if pending_approval:
            st.markdown("---")
            st.markdown(
                f"### Pending approval "
                f"({len(pending_approval)} event(s) awaiting CHRO sign-off)"
            )
            for e in pending_approval:
                with st.expander(f"{e.event_type} — {e.event_id} "
                                   f"(by {e.user_id})"):
                    st.json(e.to_dict())
                    if st.button(f"Approve {e.event_id}",
                                   key=f"approve_{e.event_id}"):
                        store.approve(e.event_id, approved_by=user_id or "demo_chro")
                        st.success("Approved.")
                        st.rerun()


# ─── SIDEBAR ────────────────────────────────────────────────────────
def _industry_selector_sidebar() -> str:
    """Sidebar selector for the active industry vertical.

    On change:
      1. Stores the new key in st.session_state["industry_key"]
      2. Calls set_industry(REGISTRY[key]) so the tool layer sees it
      3. Returns the key — main() uses it to route load_data(industry_key)
         which, because @st.cache_data is keyed on the argument, picks up
         the new vertical's CSVs without needing a manual cache clear.

    Byte-identity contract: the default is "energy" so a fresh session
    loads identically to pre-v6 behavior (no change until user toggles).
    """
    if "industry_key" not in st.session_state:
        st.session_state["industry_key"] = "energy"

    # Display names come from the profile's display_name field.
    options = list(REGISTRY.keys())
    def _fmt(k):
        return REGISTRY[k].display_name if k in REGISTRY else k

    idx = options.index(st.session_state["industry_key"]) \
        if st.session_state["industry_key"] in options else 0

    selected = st.sidebar.selectbox(
        "INDUSTRY",
        options,
        index=idx,
        format_func=_fmt,
        help=(
            "Active industry vertical. Every downstream tool "
            "(forecast_tools, talent_tools, brain) reads dept economics, "
            "interventions, thresholds, and taxonomy from the selected "
            "profile. Workforce data is loaded from the profile's own "
            "CSV directory."
        ),
    )
    # Apply the selection to both session state and the global active profile.
    if selected != st.session_state["industry_key"]:
        st.session_state["industry_key"] = selected
    _set_industry(REGISTRY[selected])

    # Status pill — shows the active vertical at a glance.
    profile = REGISTRY[selected]
    n_depts = len(profile.dept_economics)
    st.sidebar.markdown(
        f'<div style="font-size:0.72rem;color:#6E6E73;padding:4px 0">'
        f'Active: <strong>{profile.display_name}</strong> '
        f'<span style="color:#86868B">({n_depts} depts)</span></div>',
        unsafe_allow_html=True,
    )
    return selected


def render_sidebar():
    with st.sidebar:
        st.markdown('''<div style="padding:8px 0 20px 0">
            <div style="font-size:1.25rem;font-weight:800;letter-spacing:-0.03em" class="gradient-title">
                Workforce Intelligence</div>
            <div style="font-size:0.72rem;color:#86868B;letter-spacing:0.02em;margin-top:2px">
                Multi-Agent AI System</div>
        </div>''', unsafe_allow_html=True)
        # Industry selector — drives data routing + tool profile. Added v6.
        industry_key = _industry_selector_sidebar()
        page = st.selectbox("NAVIGATION",
            ["Overview", "Talent Intelligence", "Workforce Forecasting",
             "Boardroom Brief", "Human-in-the-Lead",
             "Executive Dashboard"])
        # Learning-engine toggle lives in the sidebar so every page sees
        # the same state. Stored in session_state["learning_enabled"].
        _engine_toggle_sidebar()
        st.markdown("")
        st.markdown("---")
        st.markdown('''<div style="padding:8px 0;font-size:0.72rem;color:#86868B;line-height:1.8">
            <strong style="color:#6E6E73">Deep Learning</strong><br>
            DistilBERT &middot; GLiNER &middot; ModernBERT &middot; SBERT &middot; Bi-LSTM<br><br>
            <strong style="color:#6E6E73">Agent Brain</strong><br>
            Proprietary Brain (6 scenarios) &middot; CrewAI+Claude (Joint Hire only)<br><br>
            <strong style="color:#6E6E73">Evaluation</strong><br>
            Canonical adjudicated dev sidecar (840 spans / 25 docs)<br><br>
            <strong style="color:#6E6E73">Course</strong><br>
            ITAI 2376 &middot; Spring 2026<br>
            Jiri Musil
        </div>''', unsafe_allow_html=True)
        return page, industry_key


# ─── MAIN ───────────────────────────────────────────────────────────
def main():
    inject_css()
    page, industry_key = render_sidebar()
    try:
        employees, monthly, individual = load_data(industry_key)
    except FileNotFoundError:
        ddir = _industry_data_dir(industry_key)
        st.error(
            f"Workforce CSVs for `{industry_key}` not found at `{ddir}/`. "
            f"For Energy: run `python main.py --mode train`. "
            f"For other verticals: run `python -c \"from data.generate_workforce_data "
            f"import save_datasets_for_industry; save_datasets_for_industry('{industry_key}')\"`."
        )
        return
    if page == "Overview": render_overview(employees, monthly, individual)
    elif page == "Talent Intelligence": render_talent_page(employees, monthly)
    elif page == "Workforce Forecasting": render_forecast_page(monthly, individual)
    elif page == "Boardroom Brief": render_boardroom_brief_page(employees, monthly, individual)
    elif page == "Human-in-the-Lead": render_human_in_the_lead_page(employees, monthly, individual)
    elif page == "Executive Dashboard": render_unified(employees, monthly, individual)

if __name__ == "__main__":
    main()
