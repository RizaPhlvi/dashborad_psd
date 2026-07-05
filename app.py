import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Plantation Intelligence Dashboard",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 PLANTATION PREMIUM THEME CSS — REDESIGNED
# =========================================================
st.markdown("""
<style>
/* ============================================
   0. IMPORTS & ROOT VARIABLES
   ============================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800;900&display=swap');

:root {
    --bg-deep: #050c08;
    --bg-primary: #081a0e;
    --bg-secondary: #0c2214;
    --bg-card: rgba(12, 34, 20, 0.7);
    --bg-glass: rgba(16, 42, 26, 0.45);
    --border-subtle: rgba(74, 222, 128, 0.08);
    --border-glow: rgba(74, 222, 128, 0.2);
    --green-primary: #22c55e;
    --green-light: #4ade80;
    --green-bright: #86efac;
    --green-dark: #166534;
    --gold-primary: #d4a017;
    --gold-light: #fbbf24;
    --text-primary: #f0fdf4;
    --text-secondary: #a7c4b5;
    --text-muted: #6b8f7b;
    --shadow-green: rgba(34, 197, 94, 0.15);
    --shadow-gold: rgba(212, 160, 23, 0.15);
}

/* ============================================
   1. GLOBAL RESET & BASE
   ============================================ */
.stApp {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Animated organic background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(22, 101, 52, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(34, 197, 94, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(212, 160, 23, 0.04) 0%, transparent 50%),
        linear-gradient(180deg, #050c08 0%, #081a0e 40%, #0c2214 100%);
    z-index: -1;
    pointer-events: none;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--green-dark), var(--green-primary));
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--green-light); }

/* ============================================
   2. ANIMATED FLOATING PARTICLES (Leaves)
   ============================================ */
@keyframes floatUp {
    0% { transform: translateY(100vh) rotate(0deg) scale(0.5); opacity: 0; }
    10% { opacity: 0.6; }
    90% { opacity: 0.3; }
    100% { transform: translateY(-10vh) rotate(360deg) scale(1); opacity: 0; }
}

@keyframes gentlePulse {
    0%, 100% { opacity: 0.6; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.02); }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(74, 222, 128, 0.1); box-shadow: 0 0 15px rgba(34, 197, 94, 0.05); }
    50% { border-color: rgba(74, 222, 128, 0.25); box-shadow: 0 0 25px rgba(34, 197, 94, 0.12); }
}

@keyframes slideInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInScale {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ============================================
   3. HERO STRIP — CINEMATIC PLANTATION HEADER
   ============================================ */
.hero-strip {
    background:
        linear-gradient(135deg, rgba(5, 12, 8, 0.3) 0%, rgba(22, 101, 52, 0.4) 40%, rgba(34, 197, 94, 0.2) 100%),
        linear-gradient(225deg, rgba(212, 160, 23, 0.08) 0%, transparent 50%),
        linear-gradient(180deg, #0a1f12 0%, #143d24 50%, #1a5c35 100%);
    background-size: 200% 200%;
    animation: gradientShift 12s ease infinite;
    padding: 2.8rem 3rem;
    border-radius: 24px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(74, 222, 128, 0.15);
    box-shadow:
        0 25px 60px rgba(0, 0, 0, 0.5),
        0 0 80px rgba(34, 197, 94, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    animation: slideInUp 0.8s ease-out, gradientShift 12s ease infinite;
}

.hero-strip::before {
    content: '🌿';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 11rem;
    opacity: 0.05;
    filter: blur(2px);
    animation: gentlePulse 6s ease-in-out infinite;
}

.hero-strip::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 40px,
            rgba(74, 222, 128, 0.015) 40px,
            rgba(74, 222, 128, 0.015) 41px
        ),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 40px,
            rgba(74, 222, 128, 0.01) 40px,
            rgba(74, 222, 128, 0.01) 41px
        );
    pointer-events: none;
    z-index: 1;
}

.hero-title {
    font-family: 'Playfair Display', 'Inter', serif;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #86efac 50%, #fbbf24 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 5s linear infinite;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 2;
    letter-spacing: -0.02em;
}

.hero-accent {
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, #4ade80, #22c55e, #d4a017, #fbbf24);
    background-size: 200% auto;
    animation: shimmer 3s linear infinite;
    border-radius: 4px;
    margin-bottom: 1rem;
    position: relative;
    z-index: 2;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255, 255, 255, 0.82);
    line-height: 1.7;
    max-width: 80%;
    margin-bottom: 1.4rem;
    position: relative;
    z-index: 2;
    font-weight: 400;
}

.hero-badges {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}

.hero-badge {
    padding: 0.45rem 1rem;
    border-radius: 999px;
    background: rgba(74, 222, 128, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(74, 222, 128, 0.25);
    font-size: 0.82rem;
    font-weight: 600;
    color: #bbf7d0;
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.08);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.hero-badge:hover {
    background: rgba(74, 222, 128, 0.2);
    border-color: rgba(74, 222, 128, 0.5);
    box-shadow: 0 0 30px rgba(74, 222, 128, 0.2);
    transform: translateY(-2px);
}

.hero-mini-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    z-index: 2;
    flex-wrap: wrap;
}

.mini-stat-label {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

.mini-stat-value {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
}

/* ============================================
   4. SIDEBAR — ORGANIC DARK PANEL
   ============================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c08 0%, #081a0e 50%, #0a1f12 100%) !important;
    border-right: 1px solid rgba(74, 222, 128, 0.08) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

.sidebar-block {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.sidebar-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74, 222, 128, 0.3), transparent);
}

.sidebar-block:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 32px rgba(34, 197, 94, 0.08);
}

.sidebar-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--green-light);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.commodity-brief {
    background: rgba(34, 197, 94, 0.08);
    backdrop-filter: blur(8px);
    border-left: 3px solid var(--green-light);
    padding: 1rem;
    border-radius: 0 14px 14px 0;
    margin-top: 0.6rem;
    transition: all 0.3s ease;
}

.commodity-brief:hover {
    background: rgba(34, 197, 94, 0.14);
    transform: translateX(4px);
}

.active-filter-box {
    background: rgba(212, 160, 23, 0.08);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(212, 160, 23, 0.2);
    border-radius: 14px;
    padding: 0.9rem;
    font-size: 0.82rem;
    color: #fef3c7;
    margin-top: 0.6rem;
}

/* ============================================
   5. INTEL KPI CARDS — GLASSMORPHISM + GLOW
   ============================================ */
.intel-kpi {
    background:
        linear-gradient(145deg, rgba(12, 34, 20, 0.8) 0%, rgba(8, 26, 14, 0.9) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 1.4rem 1.3rem;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
    animation: fadeInScale 0.6s ease-out;
}

.intel-kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--green-light), transparent);
    opacity: 0;
    transition: opacity 0.4s ease;
}

.intel-kpi::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(74, 222, 128, 0.03) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}

.intel-kpi:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow:
        0 20px 50px rgba(0, 0, 0, 0.4),
        0 0 40px var(--shadow-green);
    border-color: var(--border-glow);
}

.intel-kpi:hover::before { opacity: 1; }
.intel-kpi:hover::after { opacity: 1; }

.kpi-layer1 {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.kpi-layer2 {
    font-size: 1.9rem;
    font-weight: 900;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 0.35rem;
    letter-spacing: -0.02em;
    text-shadow: 0 0 30px rgba(74, 222, 128, 0.15);
}

.kpi-layer3 {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    font-weight: 400;
}

/* ============================================
   6. SECTION TITLES — ORGANIC UNDERLINE
   ============================================ */
.section-title {
    font-family: 'Playfair Display', 'Inter', serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #bbf7d0 0%, #4ade80 50%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 1.8rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, rgba(74, 222, 128, 0.4), rgba(212, 160, 23, 0.2), transparent) 1;
    display: inline-block;
    letter-spacing: -0.01em;
}

.section-subtitle {
    font-size: 0.92rem;
    color: var(--text-muted);
    margin-bottom: 1.4rem;
    font-style: italic;
    padding-left: 0.5rem;
    border-left: 2px solid rgba(74, 222, 128, 0.2);
}

/* ============================================
   7. INSIGHT / WATCHLIST / REC CARDS
   ============================================ */
.insight-card {
    background:
        linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(74, 222, 128, 0.04) 100%);
    backdrop-filter: blur(12px);
    border-left: 4px solid var(--green-light);
    padding: 1.2rem 1.5rem;
    border-radius: 4px 18px 18px 4px;
    color: #d1fae5;
    margin: 1rem 0;
    box-shadow:
        0 8px 24px rgba(34, 197, 94, 0.08),
        inset 0 0 30px rgba(34, 197, 94, 0.03);
    line-height: 1.7;
    transition: all 0.3s ease;
    position: relative;
}

.insight-card:hover {
    transform: translateX(6px);
    box-shadow: 0 12px 32px rgba(34, 197, 94, 0.15);
    border-left-width: 6px;
}

.watchlist-card {
    background:
        linear-gradient(135deg, rgba(212, 160, 23, 0.1) 0%, rgba(251, 191, 36, 0.04) 100%);
    backdrop-filter: blur(12px);
    border-left: 4px solid var(--gold-primary);
    padding: 1.2rem 1.5rem;
    border-radius: 4px 18px 18px 4px;
    color: #fef3c7;
    margin: 1rem 0;
    line-height: 1.7;
    box-shadow: 0 8px 24px rgba(212, 160, 23, 0.08);
    transition: all 0.3s ease;
}

.watchlist-card:hover {
    transform: translateX(6px);
    box-shadow: 0 12px 32px rgba(212, 160, 23, 0.15);
}

.rec-card {
    background:
        linear-gradient(135deg, rgba(132, 204, 22, 0.1) 0%, rgba(163, 230, 53, 0.04) 100%);
    backdrop-filter: blur(12px);
    border-left: 4px solid #84cc16;
    padding: 1.2rem 1.5rem;
    border-radius: 4px 18px 18px 4px;
    color: #ecfccb;
    margin: 0.7rem 0;
    line-height: 1.7;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px rgba(132, 204, 22, 0.06);
}

.rec-card:hover {
    transform: translateX(6px);
    box-shadow: 0 10px 28px rgba(132, 204, 22, 0.12);
}

.priority-card {
    background:
        linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(125, 211, 252, 0.03) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-top: 3px solid #38bdf8;
    padding: 1.2rem 1.4rem;
    border-radius: 16px;
    color: #e0f2fe;
    margin: 0.6rem 0;
    transition: all 0.4s ease;
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.06);
}

.priority-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.3);
}

/* ============================================
   8. TABS — ORGANIC PILL STYLE
   ============================================ */
button[data-baseweb="tab"] {
    border-radius: 14px !important;
    font-weight: 600 !important;
    background: rgba(12, 34, 20, 0.6) !important;
    backdrop-filter: blur(8px) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 0.75rem 1.4rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-size: 0.88rem !important;
}

button[data-baseweb="tab"]:hover {
    background: rgba(34, 197, 94, 0.1) !important;
    border-color: var(--border-glow) !important;
    color: var(--green-light) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #166534 0%, #22c55e 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(74, 222, 128, 0.4) !important;
    box-shadow:
        0 4px 20px rgba(34, 197, 94, 0.3),
        0 0 40px rgba(34, 197, 94, 0.1) !important;
}

/* ============================================
   9. FORM CONTROLS
   ============================================ */
.stSelectbox label, .stSlider label, .stRadio label,
.stMultiSelect label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

/* Radio buttons styling */
div[data-testid="stRadio"] label {
    border-radius: 12px;
    padding: 0.5rem 0.8rem;
    transition: all 0.2s ease;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(74, 222, 128, 0.06);
}

/* ============================================
   10. DATAFRAMES & METRICS
   ============================================ */
div[data-testid="stMetric"] { display: none; }

header[data-testid="stHeader"] {
    background: rgba(5, 12, 8, 0.8) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-subtle);
}

.stDataFrame {
    border-radius: 16px !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden;
}

/* ============================================
   11. BUTTONS
   ============================================ */
.stButton > button {
    border-radius: 14px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #166534, #22c55e) !important;
    color: white !important;
    border: 1px solid rgba(74, 222, 128, 0.3) !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.15) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(34, 197, 94, 0.3) !important;
}

/* Download buttons */
.stDownloadButton > button {
    border-radius: 14px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, rgba(212, 160, 23, 0.2), rgba(251, 191, 36, 0.1)) !important;
    color: #fef3c7 !important;
    border: 1px solid rgba(212, 160, 23, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, rgba(212, 160, 23, 0.35), rgba(251, 191, 36, 0.2)) !important;
    box-shadow: 0 8px 25px rgba(212, 160, 23, 0.2) !important;
    transform: translateY(-2px) !important;
}

/* ============================================
   12. WARNING / INFO BOXES
   ============================================ */
.stWarning {
    border-radius: 14px !important;
    border-left: 4px solid #fbbf24 !important;
    background: rgba(251, 191, 36, 0.08) !important;
    backdrop-filter: blur(8px) !important;
}

/* ============================================
   13. PLOTLY CHART CONTAINERS
   ============================================ */
.js-plotly-plot .plotly {
    border-radius: 16px;
    overflow: hidden;
}

/* ============================================
   14. ORGANIC DIVIDER
   ============================================ */
.organic-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(74, 222, 128, 0.2), rgba(212, 160, 23, 0.15), transparent);
    margin: 2rem 0;
    border-radius: 2px;
}

/* ============================================
   15. CODE BLOCKS (Decision Tree)
   ============================================ */
.stCodeBlock {
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    background: rgba(8, 26, 14, 0.8) !important;
}

/* ============================================
   16. RESPONSIVE ADJUSTMENTS
   ============================================ */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .hero-strip { padding: 1.8rem 1.5rem; }
    .hero-mini-stats { gap: 1rem; }
    .kpi-layer2 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET EMBEDDED (BPS 2024)
# =========================================================
CSV_DATA = """Provinsi,Kelapa Sawit,Kelapa,Karet,Kopi,Kakao,Teh,Tebu
ACEH,1092.71,64.1,51.17,74.13,34.17,0,0
SUMATERA UTARA,5120.02,103.64,251.52,91.69,38.48,10.14,14.52
SUMATERA BARAT,1410.64,79.06,99.67,15.32,34.58,5.6,0
RIAU,9136.1,399.95,180.74,1.83,0.84,0,0
JAMBI,2113.97,115.35,222.56,23.23,0.69,4.24,0
SUMATERA SELATAN,3967.39,61.65,619.55,219.59,4.4,2.62,124.37
BENGKULU,1298.17,7.25,64.6,55.63,2.36,1.88,0
LAMPUNG,375.24,83.76,84.83,120.38,48.11,0,644.48
KEP. BANGKA BELITUNG,860.24,4.55,26.04,0.11,0.29,0,0
KEP. RIAU,24.19,12.16,8.41,0,0.01,0,0
DKI JAKARTA,0,0,0,0,0,0,0
JAWA BARAT,46.75,88.14,21.17,26.48,0.71,80.24,55.17
JAWA TENGAH,0,139.01,20.73,26.79,1.31,11.81,250.07
DI YOGYAKARTA,0,50.08,0.01,1.88,2.03,0.23,3.22
JAWA TIMUR,0,194.14,16.27,53.25,10.56,2.14,1252.84
BANTEN,32.84,45.83,5.77,2.02,2.12,0.01,0
BALI,0,68.39,0,14.71,4.87,0,0
NUSA TENGGARA BARAT,0,49.81,0,6.42,2.59,0,18.22
NUSA TENGGARA TIMUR,0,62.15,0,24.38,21.08,0,10.77
KALIMANTAN BARAT,4958.54,76.8,158.41,2.98,0.6,0,0
KALIMANTAN TENGAH,7458.14,16.6,108.29,0.24,1.61,0,0
KALIMANTAN SELATAN,1255.08,24.01,127.96,0.89,0.05,0,0
KALIMANTAN TIMUR,3905.19,9.82,53.49,0.12,1.03,0,0
KALIMANTAN UTARA,611.14,0.64,0.16,0.11,0.79,0,0
SULAWESI UTARA,0,270.3,0,3.72,5.62,0,0
SULAWESI TENGAH,384.2,199.64,1.37,3.13,125.2,0,0
SULAWESI SELATAN,133.5,67.57,3.77,31.79,80.52,0,22.56
SULAWESI TENGGARA,75.47,42.87,0.22,2.61,98.02,0,15.41
GORONTALO,22.83,66.7,0,0.13,1.54,0,53.89
SULAWESI BARAT,366.68,36.72,0,4.75,67.14,0,0
MALUKU,22.33,107.73,0.6,0.49,8.59,0,0
MALUKU UTARA,20.14,204.17,0,0.02,7.38,0,0
PAPUA BARAT,40.38,2.02,0,0.01,0.24,0,0
PAPUA BARAT DAYA,43.44,14.35,0,0,0.76,0,0
PAPUA,130.55,9.77,0,0.09,8,0,0
PAPUA SELATAN,482.37,4.35,4.82,0,0.01,0,0
PAPUA TENGAH,47.98,1.2,0,1.18,0.82,0,0
PAPUA PEGUNUNGAN,0,0.02,0,3.27,0,0,0"""

@st.cache_data
def load_data():
    return pd.read_csv(io.StringIO(CSV_DATA))

df = load_data()
numeric_cols = [c for c in df.columns if c != "Provinsi"]

# =========================================================
# COMMODITY IDENTITY MAP
# =========================================================
COMMODITY_IDENTITY = {
    "Kelapa Sawit": {"icon": "🌴", "color": "#166534", "color_light": "#4ade80", "sector": "Perkebunan Besar", "desc": "Primadona ekspor. Mendominasi Sumatera & Kalimantan."},
    "Kelapa": {"icon": "🥥", "color": "#0369a1", "color_light": "#38bdf8", "sector": "Perkebunan Rakyat", "desc": "Tanaman pesisir Nusantara. Potensi hilirisasi minyak & sabut."},
    "Karet": {"icon": "🌳", "color": "#78350f", "color_light": "#c7815e", "sector": "Perkebunan Campuran", "desc": "Sentra di Sumatera. Tantangan fluktuasi harga & peremajaan."},
    "Kopi": {"icon": "☕", "color": "#78350f", "color_light": "#d4a574", "sector": "Perkebunan Rakyat", "desc": "Ikon specialty coffee dunia (Gayo, Toraja, Kintamani)."},
    "Kakao": {"icon": "🍫", "color": "#451a03", "color_light": "#a0522d", "sector": "Perkebunan Rakyat", "desc": "Sulawesi sebagai tulang punggung. Perlu peremajaan pohon."},
    "Teh": {"icon": "🍃", "color": "#064e3b", "color_light": "#34d399", "sector": "Perkebunan Besar", "desc": "Eksklusif dataran tinggi dingin (Jabar & Sumut)."},
    "Tebu": {"icon": "🌾", "color": "#3f6212", "color_light": "#84cc16", "sector": "Perkebunan Strategis", "desc": "Bahan baku gula nasional (Jatim & Lampung)."}
}

def get_comm_attr(comm, attr="color"):
    return COMMODITY_IDENTITY.get(comm, {}).get(attr, "#4ade80")

def format_num(x):
    try:
        v = float(x)
        if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
        if v >= 1_000: return f"{v/1_000:.1f}K"
        return f"{v:,.0f}"
    except: return str(x)

def format_ton(x):
    try: return f"{float(x):,.2f}"
    except: return str(x)

def create_intel_kpi(label, value, subtext, icon, color):
    return f'''<div class="intel-kpi" style="border-top: 3px solid {color};">
        <div class="kpi-layer1">{icon} {label}</div>
        <div class="kpi-layer2">{value}</div>
        <div class="kpi-layer3">{subtext}</div>
    </div>'''

def apply_plantation_layout(fig, height=480):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(8, 26, 14, 0.6)",
        plot_bgcolor="rgba(12, 34, 20, 0.8)",
        font=dict(color="#c7d2c0", family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=40, r=30, t=60, b=45),
        xaxis=dict(
            gridcolor="rgba(74, 222, 128, 0.06)",
            zerolinecolor="rgba(74, 222, 128, 0.1)",
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor="rgba(74, 222, 128, 0.06)",
            zerolinecolor="rgba(74, 222, 128, 0.1)",
            tickfont=dict(size=11)
        ),
        title=dict(
            font=dict(size=15, color="#bbf7d0", family="Inter"),
            x=0.02, xanchor="left"
        ),
        hoverlabel=dict(
            bgcolor="rgba(12, 34, 20, 0.95)",
            bordercolor="rgba(74, 222, 128, 0.3)",
            font=dict(color="#e8f0e4", family="Inter", size=12)
        )
    )
    return fig

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('''<div class="sidebar-block">
        <div class="sidebar-title">🧭 Navigasi Intelijen</div>''', unsafe_allow_html=True)

    menu = st.radio("Menu",
        ["🏠 Ringkasan Nasional", "🌴 Profil Komoditas", "🗺️ Profil Provinsi",
         "🔬 Eksplorasi Visual", "📊 Analisis Produksi", "🌍 Sebaran Wilayah",
         "📈 Proyeksi & Model", "🧠 Insight & Strategi", "📦 Data & Ekspor"],
        label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-block"><div class="sidebar-title">🎛️ Filter Aktif</div>', unsafe_allow_html=True)
    selected_commodity = st.selectbox("Komoditas Fokus", numeric_cols, index=0, key="side_comm")
    selected_province = st.selectbox("Wilayah Provinsi", ["Semua Provinsi"] + df["Provinsi"].tolist(), index=0, key="side_prov")
    top_n = st.slider("Top N Sentra", 5, 20, 10, key="side_topn")
    show_zeros = st.checkbox("Tampilkan wilayah tanpa produksi", value=True, key="side_zero")
    st.markdown('</div>', unsafe_allow_html=True)

    comm_info = COMMODITY_IDENTITY[selected_commodity]
    st.markdown(f'''<div class="sidebar-block">
        <div class="sidebar-title">🌱 Commodity Brief</div>
        <div class="commodity-brief" style="border-left-color: {comm_info["color_light"]};">
            <div style="font-size:1.3rem; margin-bottom:0.3rem;">{comm_info["icon"]} <b style="color:#f0fdf4;">{selected_commodity}</b></div>
            <div style="font-size:0.72rem; color:#6b8f7b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem;">{comm_info["sector"]}</div>
            <div style="font-size:0.82rem; color:#a7c4b5; line-height:1.5;">{comm_info["desc"]}</div>
        </div>
    </div>''', unsafe_allow_html=True)

    # Decorative leaf element at bottom of sidebar
    st.markdown('''
    <div style="text-align:center; padding:1.5rem 0 0.5rem; opacity:0.3;">
        <div style="font-size:2.5rem;">🌿</div>
        <div style="font-size:0.65rem; color:#6b8f7b; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.3rem;">Plantation Intel</div>
    </div>
    ''', unsafe_allow_html=True)

active_df = df.copy()
if selected_province != "Semua Provinsi":
    active_df = active_df[active_df["Provinsi"] == selected_province].copy()
if not show_zeros:
    active_df = active_df[(active_df[numeric_cols].sum(axis=1) > 0)].copy()

# =========================================================
# PAGE 1: RINGKASAN NASIONAL
# =========================================================
if menu == "🏠 Ringkasan Nasional":
    total_prod = active_df[numeric_cols].sum().sum()
    active_provs = len(active_df)

    st.markdown(f"""
    <div class="hero-strip">
        <div class="hero-title">🌴 Plantation Intelligence Dashboard</div>
        <div class="hero-accent"></div>
        <div class="hero-subtitle">Pusat intelijen strategis untuk pemetaan sentra produksi, portofolio komoditas andalan, dan analisis struktur perkebunan nasional per provinsi.</div>
        <div class="hero-badges">
            <span class="hero-badge">🌾 Sentra Produksi</span>
            <span class="hero-badge">📊 Portofolio Komoditas</span>
            <span class="hero-badge">🗺️ Intelijen Wilayah</span>
            <span class="hero-badge">📈 Proyeksi Panen</span>
        </div>
        <div class="hero-mini-stats">
            <div><div class="mini-stat-label">Provinsi</div><div class="mini-stat-value">{active_provs}</div></div>
            <div><div class="mini-stat-label">Komoditas</div><div class="mini-stat-value">7</div></div>
            <div><div class="mini-stat-label">Tahun Data</div><div class="mini-stat-value">2024</div></div>
            <div><div class="mini-stat-label">Total Output</div><div class="mini-stat-value">{format_num(total_prod)} Ton</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    comm_totals = active_df[numeric_cols].sum().sort_values(ascending=False)
    dom_comm = comm_totals.index[0] if not comm_totals.empty else "-"
    dom_val = comm_totals.iloc[0] if not comm_totals.empty else 0
    dom_share = (dom_val / total_prod * 100) if total_prod > 0 else 0

    if not active_df.empty:
        top_prov = active_df.loc[active_df[numeric_cols].sum(axis=1).idxmax(), "Provinsi"]
        top_val = active_df[numeric_cols].sum(axis=1).max()
        diverse_idx = (active_df[numeric_cols] > 0).sum(axis=1).idxmax()
        diverse_prov = active_df.loc[diverse_idx, "Provinsi"]
        diverse_count = int((active_df.loc[diverse_idx, numeric_cols] > 0).sum())
        top5_share = (active_df[numeric_cols].sum(axis=1).nlargest(5).sum() / max(1, total_prod) * 100)
    else:
        top_prov = diverse_prov = "-"
        top_val = diverse_count = top5_share = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(create_intel_kpi("Komoditas Dominan", dom_comm, f"Menyumbang {dom_share:.1f}%", get_comm_attr(dom_comm,"icon"), get_comm_attr(dom_comm,"color_light")), unsafe_allow_html=True)
    with c2: st.markdown(create_intel_kpi("Sentra Tertinggi", top_prov[:14], f"Output {format_ton(top_val)}", "🏆", "#d4a017"), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Terdiversifikasi", diverse_prov[:14], f"{diverse_count} komoditas", "🌱", "#4ade80"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Konsentrasi Top-5", f"{top5_share:.1f}%", "Pangsa 5 provinsi", "🎯", "#38bdf8"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌾 Sentra & Kontribusi Produksi Nasional</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        top_df = active_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(top_n)
        fig1 = px.bar(top_df[::-1], x=selected_commodity, y="Provinsi", orientation='h', text=selected_commodity,
                      title=f"Sentra Produksi: {selected_commodity}", color_discrete_sequence=[get_comm_attr(selected_commodity, "color_light")])
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig1, 520), use_container_width=True, key="nat_bar_01")
    with col_r:
        prov_totals = active_df.copy()
        prov_totals["Total_Output"] = prov_totals[numeric_cols].sum(axis=1)
        top_prov_df = prov_totals.nlargest(top_n, "Total_Output")[["Provinsi", "Total_Output"]].sort_values("Total_Output", ascending=True)
        fig2 = px.bar(top_prov_df, x="Total_Output", y="Provinsi", orientation='h', text="Total_Output",
                      title="Kontribusi Output Perkebunan per Provinsi", color_discrete_sequence=["#d4a017"])
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig2, 520), use_container_width=True, key="nat_bar_02")

    st.markdown('<div class="organic-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ Peta Struktur Komoditas Nasional</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        comp_df = active_df[numeric_cols].sum().reset_index()
        comp_df.columns = ["Komoditas", "Produksi"]
        fig_pie = px.pie(comp_df, values="Produksi", names="Komoditas", hole=0.55,
                         color="Komoditas", color_discrete_map={c: get_comm_attr(c, "color_light") for c in comp_df["Komoditas"]},
                         title="Komposisi Output Nasional")
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(apply_plantation_layout(fig_pie, 450), use_container_width=True, key="nat_pie_01")
    with col_t2:
        tm_df = active_df.copy()
        tm_df["Total"] = tm_df[numeric_cols].sum(axis=1)
        tm_df = tm_df[tm_df["Total"] > 0].nlargest(15, "Total")
        fig_tm = px.treemap(tm_df, path=["Provinsi"], values="Total", color="Total", color_continuous_scale="YlGn",
                            title="15 Wilayah Teratas (Treemap)")
        st.plotly_chart(apply_plantation_layout(fig_tm, 450), use_container_width=True, key="nat_tree_01")

    st.markdown('<div class="section-title">⚠️ Watchlist Strategis</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="watchlist-card">
        <b>📊 Poin Kritis:</b><br>
        • <b>Konsentrasi Spasial:</b> Top 5 provinsi menguasai <b>{top5_share:.1f}%</b> — mitigasi risiko iklim & pasar.<br>
        • <b>Dominasi {dom_comm}:</b> Penopang devisa, namun rentan fluktuasi harga global.<br>
        • <b>Model Diversifikasi:</b> <b>{diverse_prov}</b> menunjukkan portofolio seimbang.<br>
        • <b>Kesenjangan:</b> Teh & Tebu sebaran sangat sempit, butuh klasterisasi spesifik lokasi.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 2: PROFIL KOMODITAS
# =========================================================
elif menu == "🌴 Profil Komoditas":
    target = st.selectbox("Pilih Komoditas", numeric_cols, key="prof_comm_sel")
    info = COMMODITY_IDENTITY[target]

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(12,34,20,0.8) 0%, rgba(8,26,14,0.9) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(74,222,128,0.1); border-radius: 20px; padding: 2rem; margin-bottom: 1.8rem; position: relative; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.3);">
        <div style="position:absolute; top:-20px; right:-20px; font-size:8rem; opacity:0.04; transform:rotate(15deg);">{info['icon']}</div>
        <div style="font-family:'Playfair Display',serif; font-size:2rem; font-weight:800; background: linear-gradient(135deg, #bbf7d0, {info['color_light']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.6rem;">{info['icon']} Profil Komoditas: {target}</div>
        <div style="font-size:0.95rem; color:#a7c4b5; line-height:1.7;"><b style="color:#c7d2c0;">Sektor:</b> {info['sector']}<br><b style="color:#c7d2c0;">Potret:</b> {info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    c_df = active_df[["Provinsi", target]].copy()
    c_df = c_df[c_df[target] > 0].sort_values(target, ascending=False)
    tot = c_df[target].sum()
    top_p = c_df.iloc[0]["Provinsi"] if not c_df.empty else "-"
    top_v = c_df.iloc[0][target] if not c_df.empty else 0
    med = c_df[target].median() if not c_df.empty else 0
    share5 = c_df.head(5)[target].sum()/max(1,tot)*100 if not c_df.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_intel_kpi("Total Panen", format_num(tot), "Volume agregat", info["icon"], info["color_light"]), unsafe_allow_html=True)
    with c2: st.markdown(create_intel_kpi("Sentra Utama", top_p[:14], "Kontributor terbesar", "🏆", "#d4a017"), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Median Produksi", format_ton(med), "Tendensi sentral", "⚖️", "#38bdf8"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Konsentrasi Top-5", f"{share5:.1f}%", "Pangsa 5 provinsi", "🎯", "#4ade80"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Hirarki Sentra", "📊 Potret Distribusi", "🗺️ Peta Kontribusi"])
    comm_color = get_comm_attr(target, "color_light")
    with tab1:
        fig = px.bar(c_df.head(top_n)[::-1], x=target, y="Provinsi", orientation='h', text=target,
                     title=f"Hirarki Sentra Produksi: {target}", color_discrete_sequence=[comm_color])
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig, 600), use_container_width=True, key=f"comm_bar_{target}")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(c_df, x=target, nbins=15, title="Distribusi Volume", color_discrete_sequence=[comm_color])
            st.plotly_chart(apply_plantation_layout(fig_h, 480), use_container_width=True, key=f"comm_hist_{target}")
        with c2:
            fig_b = px.box(c_df, y=target, points="all", title="Persebaran & Sentra Ekstrem", color_discrete_sequence=[comm_color])
            st.plotly_chart(apply_plantation_layout(fig_b, 480), use_container_width=True, key=f"comm_box_{target}")
    with tab3:
        fig_t = px.treemap(c_df.head(15), path=["Provinsi"], values=target, color=target,
                           color_continuous_scale=[info["color"], info["color_light"]], title="Treemap Kontribusi Wilayah")
        st.plotly_chart(apply_plantation_layout(fig_t, 500), use_container_width=True, key=f"comm_tree_{target}")
        fig_p = px.pie(c_df.head(10), values=target, names="Provinsi", hole=0.5, title="Komposisi 10 Wilayah Teratas",
                       color_discrete_sequence=px.colors.sequential.YlGn[2:])
        fig_p.update_traces(textinfo='percent+label')
        st.plotly_chart(apply_plantation_layout(fig_p, 450), use_container_width=True, key=f"comm_pie_{target}")

    conc_note = "terkonsentrasi kuat" if share5 > 60 else "relatif terdistribusi"
    st.markdown(f"""
    <div class="insight-card" style="border-left-color: {info['color_light']};">
        {info['icon']} <b>Intelijen {target}:</b> Produksi bersifat <b>{conc_note}</b> ({share5:.1f}% dikuasai Top-5).
        <b>{top_p}</b> berperan sebagai anchor. Potensi hilirisasi perlu difokuskan di wilayah penyangga.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 3: PROFIL PROVINSI
# =========================================================
elif menu == "🗺️ Profil Provinsi":
    target_prov = st.selectbox("Pilih Provinsi", df["Provinsi"].tolist(), key="prof_prov_sel")
    p_row = df[df["Provinsi"] == target_prov].iloc[0]
    p_data = {c: p_row[c] for c in numeric_cols}
    tot_p = sum(p_data.values())
    active_c = sum(1 for v in p_data.values() if v > 0)
    dom_c = max(p_data, key=p_data.get) if tot_p > 0 else "-"
    dom_v = p_data[dom_c] if tot_p > 0 else 0
    dom_share = (dom_v/tot_p*100) if tot_p > 0 else 0
    nat_share = (tot_p / df[numeric_cols].sum().sum() * 100) if df[numeric_cols].sum().sum() > 0 else 0

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(12,34,20,0.8) 0%, rgba(8,26,14,0.9) 100%); backdrop-filter: blur(20px); border: 1px solid rgba(74,222,128,0.1); border-radius: 20px; padding: 2rem; margin-bottom: 1.8rem; position: relative; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.3);">
        <div style="position:absolute; top:-30px; right:-10px; font-size:9rem; opacity:0.04;">🗺️</div>
        <div style="font-family:'Playfair Display',serif; font-size:2rem; font-weight:800; background: linear-gradient(135deg, #bbf7d0, #4ade80); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.6rem;">🗺️ Profil Perkebunan: {target_prov}</div>
        <div style="font-size:0.95rem; color:#a7c4b5; line-height:1.7;">Total output <b style="color:#f0fdf4;">{format_ton(tot_p)} ribu ton</b> ({nat_share:.2f}% nasional). Komoditas andalan: <b style="color:#f0fdf4;">{dom_c}</b> ({dom_share:.1f}%).</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_intel_kpi("Total Output", format_num(tot_p), "Agregat komoditas", "📦", "#4ade80"), unsafe_allow_html=True)
    with c2: st.markdown(create_intel_kpi("Komoditas Andalan", dom_c, f"Penopang ({dom_share:.0f}%)", get_comm_attr(dom_c,"icon"), get_comm_attr(dom_c,"color_light")), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Sektor Aktif", str(active_c), f"dari {len(numeric_cols)}", "🌱", "#38bdf8"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Tingkat Dominasi", f"{dom_share:.0f}%", "Konsentrasi utama", "🎯", "#d4a017"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌳 Struktur & Portofolio Komoditas</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        p_df = pd.DataFrame({"Komoditas": list(p_data.keys()), "Produksi": list(p_data.values())}).sort_values("Produksi", ascending=False)
        fig = px.bar(p_df, x="Komoditas", y="Produksi", text="Produksi", color="Komoditas",
                     color_discrete_map={c: get_comm_attr(c, "color_light") for c in p_df["Komoditas"]}, title="Portofolio Komoditas")
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig, 460), use_container_width=True, key=f"prov_bar_{target_prov}")
    with col2:
        fig2 = px.pie(p_df[p_df["Produksi"]>0], values="Produksi", names="Komoditas", hole=0.5, title="Komposisi Output Daerah",
                      color_discrete_sequence=px.colors.sequential.YlGn[2:])
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(apply_plantation_layout(fig2, 460), use_container_width=True, key=f"prov_pie_{target_prov}")

    st.markdown('<div class="organic-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Radar Portofolio vs Nasional</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        nat_avg = {c: df[c].mean() for c in numeric_cols}
        p_values = [p_data[c] for c in numeric_cols]
        p_max = max(p_values) if p_values else 1
        p_norm = [(v / p_max * 100) if p_max > 0 else 0 for v in p_values]
        nat_values = list(nat_avg.values())
        nat_max = max(nat_values) if nat_values else 1
        nat_norm = [(v / nat_max * 100) if nat_max > 0 else 0 for v in nat_values]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=p_norm + [p_norm[0]], theta=numeric_cols + [numeric_cols[0]],
                                        fill='toself', name=target_prov, line_color="#4ade80", fillcolor="rgba(74, 222, 128, 0.25)", line=dict(width=2)))
        fig_r.add_trace(go.Scatterpolar(r=nat_norm + [nat_norm[0]], theta=numeric_cols + [numeric_cols[0]],
                                        fill='toself', name="Rata-rata Nasional", line_color="#d4a017", fillcolor="rgba(212, 160, 23, 0.1)", line=dict(width=2, dash="dot")))
        fig_r.update_layout(polar=dict(bgcolor="rgba(12, 34, 20, 0.6)",
                                       radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(74, 222, 128, 0.1)"),
                                       angularaxis=dict(gridcolor="rgba(74, 222, 128, 0.1)", tickfont=dict(color="#a7c4b5"))),
                            showlegend=True, title="Radar (Normalisasi)")
        st.plotly_chart(apply_plantation_layout(fig_r, 500), use_container_width=True, key=f"prov_radar_{target_prov}")
    with col_r2:
        bench_df = pd.DataFrame({"Komoditas": numeric_cols, "Provinsi": p_values, "Nasional": list(nat_avg.values())})
        bench_df["Deviasi"] = bench_df["Provinsi"] - bench_df["Nasional"]
        colors = ["#4ade80" if v >= 0 else "#c7815e" for v in bench_df["Deviasi"]]
        fig3 = go.Figure(go.Bar(y=bench_df["Komoditas"], x=bench_df["Deviasi"], orientation='h', marker_color=colors))
        fig3.update_layout(title="Deviasi vs Rata-rata Nasional", xaxis_title="Deviasi (Ribu Ton)")
        st.plotly_chart(apply_plantation_layout(fig3, 500), use_container_width=True, key=f"prov_bench_{target_prov}")

    dep_note = "⚠️ Sangat bergantung pada satu komoditas." if dom_share > 70 else "✅ Portofolio relatif seimbang."
    st.markdown(f'<div class="insight-card">🗺️ <b>{target_prov}</b>: <b>{active_c}</b> komoditas aktif. {dep_note} Strategi: perkuat klaster unggulan & hilirisasi lokal.</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 4: EKSPLORASI VISUAL (EDA 5-TAB)
# =========================================================
elif menu == "🔬 Eksplorasi Visual":
    st.markdown('<div class="section-title">🔬 Eksplorasi Visual & EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">5 perspektif visual: Overview, Distribusi, Hubungan, Korelasi, Deep Dive</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Overview", "📊 Distribusi", "🔗 Hubungan", "🌡️ Korelasi", "📍 Deep Dive"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            top_df = active_df[["Provinsi", selected_commodity]].sort_values(selected_commodity, ascending=False).head(top_n)
            fig = px.bar(top_df[::-1], x=selected_commodity, y="Provinsi", orientation='h', text=selected_commodity,
                         title=f"Top Sentra {selected_commodity}", color_discrete_sequence=[get_comm_attr(selected_commodity,"color_light")])
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(apply_plantation_layout(fig, 500), use_container_width=True, key="eda_overview_1")
        with c2:
            temp = active_df.copy(); temp["Total"] = temp[numeric_cols].sum(axis=1)
            top_total = temp.nlargest(top_n, "Total").sort_values("Total", ascending=True)
            fig2 = px.bar(top_total, x="Total", y="Provinsi", orientation='h', text="Total", title="Top Provinsi Total Produksi", color_discrete_sequence=["#d4a017"])
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(apply_plantation_layout(fig2, 500), use_container_width=True, key="eda_overview_2")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(active_df, x=selected_commodity, nbins=15, title=f"Distribusi {selected_commodity}",
                               color_discrete_sequence=[get_comm_attr(selected_commodity,"color_light")])
            st.plotly_chart(apply_plantation_layout(fig, 480), use_container_width=True, key="eda_dist_1")
        with c2:
            melted = active_df.melt(id_vars="Provinsi", value_vars=numeric_cols, var_name="Komoditas", value_name="Produksi")
            fig2 = px.box(melted, x="Komoditas", y="Produksi", color="Komoditas", title="Persebaran Antar Komoditas",
                          color_discrete_map={c: get_comm_attr(c, "color_light") for c in numeric_cols})
            fig2.update_xaxes(tickangle=35); fig2.update_layout(showlegend=False)
            st.plotly_chart(apply_plantation_layout(fig2, 480), use_container_width=True, key="eda_dist_2")
    with tab3:
        c1, c2 = st.columns(2)
        x_v = c1.selectbox("Komoditas X", numeric_cols, index=0, key="eda_rel_x")
        y_v = c2.selectbox("Komoditas Y", numeric_cols, index=1, key="eda_rel_y")
        if x_v != y_v:
            fig = px.scatter(active_df, x=x_v, y=y_v, hover_name="Provinsi", title=f"{x_v} vs {y_v}", color_discrete_sequence=["#4ade80"])
            st.plotly_chart(apply_plantation_layout(fig, 550), use_container_width=True, key="eda_rel_scatter")
            corr = active_df[[x_v, y_v]].corr().iloc[0,1]
            strength = "kuat" if abs(corr)>0.6 else "sedang" if abs(corr)>0.3 else "lemah"
            st.markdown(f'<div class="insight-card">🔗 <b>Korelasi:</b> {corr:.3f} ({strength})</div>', unsafe_allow_html=True)
        else: st.warning("Pilih dua komoditas berbeda.")
    with tab4:
        corr_m = active_df[numeric_cols].corr()
        fig = px.imshow(corr_m, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale=[[0,"#7c4f2a"],[0.25,"#a0522d"],[0.5,"#166534"],[0.75,"#22c55e"],[1,"#4ade80"]],
                        title="Matriks Korelasi Agro-Ekologis")
        fig.update_traces(textfont=dict(color="white", size=11), xgap=3, ygap=3)
        fig.update_xaxes(side="bottom", tickangle=30); fig.update_yaxes(autorange="reversed")
        fig.update_layout(paper_bgcolor="rgba(8,26,14,0.6)", plot_bgcolor="rgba(12,34,20,0.8)",
                          font=dict(color="#c7d2c0"), margin=dict(l=40,r=30,t=60,b=40))
        fig.update_coloraxes(colorbar=dict(title=dict(text="Korelasi", font=dict(color="#c7d2c0")),
                                            tickfont=dict(color="#c7d2c0"), len=0.75))
        st.plotly_chart(fig, use_container_width=True, key="eda_corr")
    with tab5:
        deep_prov = st.selectbox("Pilih Provinsi", active_df["Provinsi"].tolist(), key="eda_deep")
        p_df = active_df[active_df["Provinsi"]==deep_prov]
        if not p_df.empty:
            row = p_df.iloc[0]
            profile = pd.DataFrame({"Komoditas": numeric_cols, "Produksi": [row[c] for c in numeric_cols]}).sort_values("Produksi", ascending=False)
            c1, c2 = st.columns([1.1, 1])
            with c1:
                fig = px.bar(profile, x="Komoditas", y="Produksi", text="Produksi", color="Komoditas",
                             color_discrete_map={c: get_comm_attr(c, "color_light") for c in numeric_cols}, title=f"Profil {deep_prov}")
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside'); fig.update_layout(showlegend=False)
                st.plotly_chart(apply_plantation_layout(fig, 500), use_container_width=True, key=f"eda_deep_bar_{deep_prov}")
            with c2:
                total_p = profile["Produksi"].sum()
                dom_comm = profile.iloc[0]["Komoditas"]
                st.markdown(create_intel_kpi("Total Produksi", format_num(total_p), "Ribu Ton", "📦", "#4ade80"), unsafe_allow_html=True)
                st.markdown(create_intel_kpi("Komoditas Dominan", dom_comm, format_ton(profile.iloc[0]['Produksi']), get_comm_attr(dom_comm,'icon'), get_comm_attr(dom_comm,'color_light')), unsafe_allow_html=True)

# =========================================================
# PAGE 5: ANALISIS PRODUKSI
# =========================================================
elif menu == "📊 Analisis Produksi":
    st.markdown('<div class="section-title">📊 Analisis Produksi & Perbandingan</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["⚔️ Duel Wilayah", "🔗 Relasi Komoditas", "🎯 Benchmarking"])
    with tab1:
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("Wilayah A", df["Provinsi"].tolist(), index=0, key="duel_a")
        p2 = c2.selectbox("Wilayah B", df["Provinsi"].tolist(), index=3, key="duel_b")
        d1, d2 = df[df["Provinsi"]==p1].iloc[0], df[df["Provinsi"]==p2].iloc[0]
        comp = pd.DataFrame({"Komoditas": numeric_cols, p1: [d1[c] for c in numeric_cols], p2: [d2[c] for c in numeric_cols]})
        fig = go.Figure()
        fig.add_trace(go.Bar(name=p1, x=comp["Komoditas"], y=comp[p1], marker_color="#4ade80"))
        fig.add_trace(go.Bar(name=p2, x=comp["Komoditas"], y=comp[p2], marker_color="#d4a017"))
        fig.update_layout(barmode='group', title=f"{p1} vs {p2}")
        st.plotly_chart(apply_plantation_layout(fig, 480), use_container_width=True, key="duel_chart")
        w1, w2 = (comp[p1] >comp[p2]).sum(), (comp[p2] >comp[p1]).sum()
        st.markdown(f'<div class="insight-card">⚔️ {p1} unggul <b>{w1}</b> komoditas | {p2} unggul <b>{w2}</b> komoditas.</div>', unsafe_allow_html=True)
    with tab2:
        c1, c2 = st.columns(2)
        x_v = c1.selectbox("X", numeric_cols, index=0, key="rel_x")
        y_v = c2.selectbox("Y", numeric_cols, index=1, key="rel_y")
        if x_v != y_v:
            fig = px.scatter(active_df, x=x_v, y=y_v, hover_name="Provinsi", title=f"{x_v} vs {y_v}", color_discrete_sequence=["#4ade80"])
            st.plotly_chart(apply_plantation_layout(fig, 500), use_container_width=True, key="rel_scatter")
            corr = active_df[[x_v, y_v]].corr().iloc[0,1]
            st.markdown(f'<div class="insight-card">🔗 Korelasi: <b>{corr:.3f}</b></div>', unsafe_allow_html=True)
        else: st.warning("Pilih dua komoditas berbeda.")
    with tab3:
        bp = st.selectbox("Wilayah Benchmark", df["Provinsi"].tolist(), key="bench_sel")
        bd = df[df["Provinsi"]==bp].iloc[0]
        bdf = pd.DataFrame({"Komoditas": numeric_cols, "Prov": [bd[c] for c in numeric_cols], "Nat": [df[c].mean() for c in numeric_cols]})
        bdf["Rasio"] = bdf["Prov"] / bdf["Nat"].replace(0, 0.01)
        fig = go.Figure(go.Bar(x=bdf["Komoditas"], y=bdf["Rasio"], marker_color=[get_comm_attr(c, "color_light") for c in bdf["Komoditas"]]))
        fig.add_hline(y=1, line_dash="dash", line_color="#d4a017")
        fig.update_layout(title=f"{bp} vs Nasional", yaxis_title="Rasio")
        st.plotly_chart(apply_plantation_layout(fig, 480), use_container_width=True, key="bench_chart")

# =========================================================
# PAGE 6: SEBARAN WILAYAH
# =========================================================
elif menu == "🌍 Sebaran Wilayah":
    st.markdown('<div class="section-title">🌍 Sebaran Wilayah Produksi</div>', unsafe_allow_html=True)
    geo_df = active_df[["Provinsi", selected_commodity]].copy().sort_values(selected_commodity, ascending=False)
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        fig1 = px.bar(geo_df.head(top_n), x="Provinsi", y=selected_commodity, color=selected_commodity, text=selected_commodity,
                      color_continuous_scale=[get_comm_attr(selected_commodity, "color"), get_comm_attr(selected_commodity, "color_light")],
                      title="Ranking Intensitas Sentra")
        fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside'); fig1.update_xaxes(tickangle=45)
        st.plotly_chart(apply_plantation_layout(fig1, 500), use_container_width=True, key="geo_rank")
    with col_r:
        bub = geo_df.head(15).copy(); bub["Rank"] = range(1, len(bub)+1)
        fig2 = px.scatter(bub, x="Rank", y=selected_commodity, size=selected_commodity, hover_name="Provinsi", text="Provinsi",
                          title="Bubble Prioritas Wilayah", color_discrete_sequence=[get_comm_attr(selected_commodity, "color_light")])
        fig2.update_traces(textposition="top center")
        st.plotly_chart(apply_plantation_layout(fig2, 500), use_container_width=True, key="geo_bub")
    total_g = geo_df[selected_commodity].sum()
    if total_g > 0:
        geo_df["Share"] = (geo_df[selected_commodity]/total_g)*100
        geo_df["Kategori"] = pd.cut(geo_df["Share"], bins=[-1,1,5,15,100], labels=["Minor", "Penyangga", "Regional", "Nasional"])
        kc = geo_df["Kategori"].value_counts()
        st.markdown('<div class="section-title">🏷️ Klasifikasi Wilayah</div>', unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        k1.markdown(create_intel_kpi("Sentra Nasional", str(kc.get("Nasional",0)), "Anchor production", "🏆", "#d4a017"), unsafe_allow_html=True)
        k2.markdown(create_intel_kpi("Sentra Regional", str(kc.get("Regional",0)), "Pillar produksi", "🌾", "#84cc16"), unsafe_allow_html=True)
        k3.markdown(create_intel_kpi("Penyangga", str(kc.get("Penyangga",0)), "Stabilisator", "🌱", "#38bdf8"), unsafe_allow_html=True)
        k4.markdown(create_intel_kpi("Minor", str(kc.get("Minor",0)), "Niche", "🍃", "#c7815e"), unsafe_allow_html=True)

# =========================================================
# PAGE 7: PROYEKSI & MODEL (4 TAB LENGKAP)
# =========================================================
elif menu == "📈 Proyeksi & Model":
    st.markdown('<div class="section-title">📈 Proyeksi & Model Produksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="watchlist-card">📌 <b>Catatan:</b> Model untuk eksplorasi pola awal, bukan prediksi final tanpa validasi time-series.</div>', unsafe_allow_html=True)

    if active_df.shape[0] < 5:
        st.warning("⚠️ Data terlalu sedikit untuk analisis model yang stabil.")
    else:
        tab_lr, tab_fc, tab_rf, tab_dt = st.tabs(["📐 Regresi Linear", "📅 Forecasting 2025", "🌲 Random Forest", "🌳 Decision Tree"])
        with tab_lr:
            c1, c2 = st.columns(2)
            x_var = c1.selectbox("Prediktor (X)", numeric_cols, index=1, key="proj_x")
            y_var = c2.selectbox("Target (Y)", numeric_cols, index=0, key="proj_y")
            if x_var != y_var:
                mdf = active_df[[x_var, y_var]].dropna()
                if len(mdf) >= 5:
                    X, y = mdf[[x_var]].values, mdf[y_var].values
                    lr = LinearRegression().fit(X, y)
                    pred = lr.predict(X)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("MAE", f"{mean_absolute_error(y, pred):.2f}")
                    k2.metric("RMSE", f"{math.sqrt(mean_squared_error(y, pred)):.2f}")
                    k3.metric("R²", f"{r2_score(y, pred):.4f}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=mdf[x_var], y=mdf[y_var], mode="markers", name="Aktual", marker=dict(color="#4ade80", size=9, symbol="circle")))
                    sort_idx = np.argsort(mdf[x_var].values)
                    fig.add_trace(go.Scatter(x=mdf[x_var].values[sort_idx], y=pred[sort_idx], mode="lines", name="Regresi", line=dict(color="#d4a017", width=3)))
                    fig.update_layout(title=f"Pola Hubungan: {x_var} → {y_var}", xaxis_title=x_var, yaxis_title=y_var)
                    st.plotly_chart(apply_plantation_layout(fig, 480), use_container_width=True, key="proj_reg")
                    st.markdown("#### 🎮 Playground Prediksi")
                    x_input = st.number_input(f"Masukkan volume {x_var} (Ribu Ton):", min_value=0.0, value=float(np.median(mdf[x_var])))
                    pred_val = lr.predict([[x_input]])[0]
                    st.markdown(f'<div class="insight-card">🌾 Prediksi <b>{y_var}</b> jika <b>{x_var} = {x_input:.2f}</b> ribu ton adalah <b>{pred_val:.2f}</b> ribu ton.</div>', unsafe_allow_html=True)
                else:
                    st.warning("Data tidak cukup untuk membangun model regresi (minimal 5 observasi).")
            else:
                st.warning("Pilih dua komoditas yang berbeda untuk analisis.")
        with tab_fc:
            st.markdown("### 📅 Simulasi Panen 2025")
            st.markdown('<div class="watchlist-card">⚠️ Dataset cross-sectional. Forecasting via simulasi growth rate.</div>', unsafe_allow_html=True)
            fc_comm = st.selectbox("Komoditas Target", numeric_cols, key="fc_comm")
            gr = st.slider("Growth Rate (%)", 1, 20, 7, key="fc_gr") / 100
            fc = active_df[["Provinsi", fc_comm]].copy()
            fc["Proyeksi_2025"] = fc[fc_comm] * (1 + gr)
            top_fc = fc.sort_values(fc_comm, ascending=False).head(top_n)
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[fc_comm], name="Panen 2024", marker_color="#c7815e"))
            fig_fc.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Proyeksi_2025"], name="Proyeksi 2025", marker_color="#4ade80"))
            fig_fc.update_layout(barmode="group", title=f"Forecasting {fc_comm}: 2024 vs 2025", xaxis_title="Provinsi", yaxis_title="Volume (Ribu Ton)")
            st.plotly_chart(apply_plantation_layout(fig_fc, 500), use_container_width=True, key="fc_chart_bar")
            c1, c2 = st.columns(2)
            c1.metric("Total 2024", format_num(fc[fc_comm].sum()))
            c2.metric("Total 2025", format_num(fc["Proyeksi_2025"].sum()), f"+{gr*100:.0f}%")
            st.dataframe(top_fc, use_container_width=True)
        with tab_rf:
            st.markdown("### 🌲 Random Forest Regression")
            tgt = st.selectbox("Target Prediksi", numeric_cols, key="rf_tgt")
            feats = [c for c in numeric_cols if c != tgt]
            rdf = active_df[feats+[tgt]].dropna()
            if len(rdf) >= 8:
                X, y = rdf[feats], rdf[tgt]
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
                y_pred = rf.predict(X_te)
                k1,k2,k3 = st.columns(3)
                k1.metric("MAE", f"{mean_absolute_error(y_te,y_pred):.2f}")
                k2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_te,y_pred)):.2f}")
                k3.metric("R²", f"{r2_score(y_te,y_pred):.4f}")
                c1, c2 = st.columns(2)
                with c1:
                    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=True)
                    fig_imp = px.bar(x=imp.values, y=imp.index, orientation='h', title="Feature Importance", color_discrete_sequence=["#4ade80"])
                    st.plotly_chart(apply_plantation_layout(fig_imp, 450), use_container_width=True, key="rf_imp")
                with c2:
                    comp_df = pd.DataFrame({"Aktual": y_te, "Prediksi": y_pred})
                    fig_sc = px.scatter(comp_df, x="Aktual", y="Prediksi", title="Aktual vs Prediksi (Test Set)", color_discrete_sequence=["#d4a017"])
                    st.plotly_chart(apply_plantation_layout(fig_sc, 450), use_container_width=True, key="rf_scatter")
                st.markdown(f'<div class="insight-card">🌱 <b>{imp.idxmax()}</b> menjadi driver utama untuk <b>{tgt}</b> (score: {imp.max():.3f}).</div>', unsafe_allow_html=True)
            else: st.warning("Data terlalu sedikit.")
        with tab_dt:
            st.markdown("### 🌳 Decision Tree Regression")
            tgt_dt = st.selectbox("Target Prediksi", numeric_cols, key="dt_tgt")
            feats_dt = [c for c in numeric_cols if c != tgt_dt]
            dt_df = active_df[feats_dt + [tgt_dt]].dropna()
            if len(dt_df) >= 8:
                X, y = dt_df[feats_dt], dt_df[tgt_dt]
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
                dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_tr, y_tr)
                y_pred = dt.predict(X_te)
                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{mean_absolute_error(y_te, y_pred):.2f}")
                k2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_te, y_pred)):.2f}")
                k3.metric("R²", f"{r2_score(y_te, y_pred):.4f}")
                imp_dt = pd.Series(dt.feature_importances_, index=feats_dt).sort_values(ascending=True)
                fig_imp_dt = px.bar(x=imp_dt.values, y=imp_dt.index, orientation='h',
                                    title="Feature Importance (Pembagi Utama)",
                                    color_discrete_sequence=["#84cc16"])
                st.plotly_chart(apply_plantation_layout(fig_imp_dt, 400), use_container_width=True, key="dt_imp")
                st.markdown("#### 🌳 Logika Percabangan Pohon Keputusan")
                st.markdown("""
                <div class="insight-card">
                Alih-alih menggunakan gambar statis yang berat,
                dashboard ini menampilkan <b>rules (aturan logika)</b> langsung dari model.
                Ini menunjukkan secara transparan bagaimana model membagi wilayah berdasarkan threshold komoditas lain.
                </div>
                """, unsafe_allow_html=True)
                tree_rules = export_text(dt, feature_names=list(feats_dt))
                st.code(tree_rules, language="text")
            else:
                st.warning("Data terlalu sedikit untuk membangun Decision Tree.")

# =========================================================
# PAGE 8: INSIGHT & STRATEGI
# =========================================================
elif menu == "🧠 Insight & Strategi":
    st.markdown('<div class="section-title">🧠 Insight & Strategi Perkebunan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Executive briefing: temuan strategis, rekomendasi kebijakan, dan prioritas pengembangan</div>', unsafe_allow_html=True)

    tot_nat = active_df[numeric_cols].sum().sum()
    comm_sums = active_df[numeric_cols].sum().sort_values(ascending=False)
    dom_c = comm_sums.index[0] if not comm_sums.empty else "-"
    top5_share = active_df[numeric_cols].sum(axis=1).nlargest(5).sum()/max(1,tot_nat)*100 if not active_df.empty else 0
    div_idx = (active_df[numeric_cols]>0).sum(axis=1).idxmax() if not active_df.empty else 0
    div_p = active_df.loc[div_idx, "Provinsi"] if not active_df.empty else "-"

    st.markdown("### 📜 Insight Strategis Nasional")
    insights = [
        f"🌴 <b>{dom_c}</b> sebagai tulang punggung sektor perkebunan nasional ({format_ton(comm_sums.iloc[0])} ribu ton).",
        f"🗺️ <b>Konsentrasi Spasial:</b> Top 5 provinsi menguasai <b>{top5_share:.1f}%</b> output nasional.",
        f"🌱 <b>{div_p}</b> sebagai model ketahanan struktural dengan portofolio terdiversifikasi.",
        f"🔗 <b>Spesialisasi Geografis:</b> Korelasi lemah antar komoditas menunjukkan mosaik keunggulan komparatif regional.",
        f"📉 <b>Kesenjangan:</b> Teh & Tebu memiliki sebaran sangat sempit, butuh strategi klasterisasi spesifik lokasi."
    ]
    for i, ins in enumerate(insights, 1):
        st.markdown(f'<div class="insight-card"><b>#{i}</b> {ins}</div>', unsafe_allow_html=True)

    st.markdown('<div class="organic-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🚀 Rekomendasi Kebijakan & Bisnis")
    recs = [
        "💼 <b>Hilirisasi Berbasis Klaster:</b> Bangun pabrik pengolahan di sentra produksi untuk nilai tambah ekspor.",
        "🛡️ <b>Mitigasi Konsentrasi:</b> Kembangkan sentra alternatif untuk distribusi risiko.",
        "🌳 <b>Program Replanting:</b> Peremajaan tanaman tua (>25 tahun) dengan bibit unggul → produktivitas +40-60%.",
        "🗺️ <b>Branding Geografis:</b> Registrasi Indikasi Geografis (Kopi Gayo, Kakao Sulawesi, Teh Jabar).",
        "📊 <b>Sistem Time-Series:</b> Transisi ke monitoring berkala untuk forecasting akurat."
    ]
    for i, rec in enumerate(recs, 1):
        st.markdown(f'<div class="rec-card"><b>#{i}</b> {rec}</div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Prioritas Pengembangan Strategis")
    p1, p2, p3 = st.columns(3)
    with p1: st.markdown('<div class="priority-card"><b>🏭 Hilirisasi & Nilai Tambah</b><br>Pengolahan CPO, karet olahan, kopi specialty.</div>', unsafe_allow_html=True)
    with p2: st.markdown('<div class="priority-card"><b>🌱 Diversifikasi & Ketahanan</b><br>Tanaman sela & agroforestri berkelanjutan.</div>', unsafe_allow_html=True)
    with p3: st.markdown('<div class="priority-card"><b>📈 Digitalisasi & Data</b><br>Precision agriculture & dashboard real-time.</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 9: DATA & EKSPOR
# =========================================================
elif menu == "📦 Data & Ekspor":
    st.markdown('<div class="section-title">📦 Data & Ekspor Perkebunan</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Observasi", len(active_df))
    c2.metric("🌾 Komoditas", len(numeric_cols))
    c3.metric("✅ Missing", int(active_df.isnull().sum().sum()))
    c4.metric("🔄 Duplikat", int(active_df.duplicated().sum()))
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Mentah", "📊 Statistik Deskriptif", "🧹 Kualitas Data"])
    with tab1: st.dataframe(active_df, use_container_width=True)
    with tab2:
        desc = active_df[numeric_cols].describe().T; desc["Range"] = desc["max"] - desc["min"]
        st.dataframe(desc, use_container_width=True)
    with tab3:
        zero_rows = (active_df[numeric_cols].sum(axis=1) == 0).sum()
        st.markdown(f"- ✅ Missing values: {int(active_df.isnull().sum().sum())}")
        st.markdown(f"- ✅ Baris duplikat: {int(active_df.duplicated().sum())}")
        st.markdown(f"- ⚠️ Wilayah tanpa produksi: {int(zero_rows)}")
        st.markdown('<div class="watchlist-card">📌 Outlier (Riau-sawit, Jatim-tebu) merepresentasikan <b>sentra produksi riil</b> BPS.</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.download_button("⬇️ Ekspor Dataset", active_df.to_csv(index=False).encode("utf-8"), "data_perkebunan.csv", "text/csv", use_container_width=True)
    with col2: st.download_button("⬇️ Ekspor Statistik", desc.to_csv().encode("utf-8"), "statistik_perkebunan.csv", "text/csv", use_container_width=True)
    st.markdown("### 📅 Ekspor Simulasi Panen 2025")
    exp_comm = st.selectbox("Komoditas", numeric_cols, key="exp_comm")
    exp_growth = st.slider("Growth rate (%)", 1, 20, 7, key="exp_growth")/100
    export_fc = active_df[["Provinsi", exp_comm]].copy()
    export_fc["Proyeksi_2025"] = export_fc[exp_comm] * (1 + exp_growth)
    export_fc["Peningkatan"] = export_fc["Proyeksi_2025"] - export_fc[exp_comm]
    st.dataframe(export_fc, use_container_width=True)
    st.download_button(f"⬇️ Ekspor Proyeksi {exp_comm} 2025", export_fc.to_csv(index=False).encode("utf-8"), f"proyeksi_{exp_comm}_2025.csv", "text/csv", use_container_width=True)

# =========================================================
# 🌿 FOOTER — REDESIGNED
# =========================================================
st.markdown("""
<div style="text-align:center; padding:2.5rem 2rem; margin-top:3rem; background: linear-gradient(135deg, rgba(8,26,14,0.9) 0%, rgba(22,101,52,0.3) 50%, rgba(8,26,14,0.9) 100%); backdrop-filter: blur(20px); border-radius: 24px; border: 1px solid rgba(74,222,128,0.1); position: relative; overflow: hidden; box-shadow: 0 -10px 40px rgba(0,0,0,0.3);">
    <div style="position:absolute; top:0; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, #4ade80, #d4a017, transparent);"></div>
    <div style="font-size:3rem; margin-bottom:0.5rem; opacity:0.6;">🌴</div>
    <h3 style="font-family:'Playfair Display',serif; font-size:1.5rem; font-weight:800; background: linear-gradient(135deg, #bbf7d0, #4ade80, #fbbf24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom:0.6rem;">Plantation Intelligence Dashboard</h3>
    <p style="color:#a7c4b5; font-size:0.9rem; margin-bottom:0.4rem;">UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar</p>
    <p style="color:#6b8f7b; font-size:0.82rem; margin-top:0.8rem;">Streamlit + Plotly + Scikit-Learn untuk perencanaan strategis sektor perkebunan Indonesia</p>
    <div style="width:60px; height:2px; background: linear-gradient(90deg, #4ade80, #d4a017); margin: 1rem auto; border-radius:2px;"></div>
    <p style="color:#4a6b58; font-size:0.75rem; letter-spacing:0.05em;">© 2026 | Sumber: BPS — Produksi Tanaman Perkebunan Menurut Provinsi, 2024</p>
</div>
""", unsafe_allow_html=True)
