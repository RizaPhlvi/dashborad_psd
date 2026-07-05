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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🎨 TROPICAL HERITAGE THEME — MAXIMUM READABILITY
# =========================================================
st.markdown("""
<style>
/* ============================================
   0. IMPORTS & VARIABLES
   ============================================ */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    /* Paper & Card backgrounds */
    --bg-paper: #FAF7F0;
    --bg-card: #FFFFFF;
    --bg-sidebar: #F0EBE0;
    --bg-sidebar-header: #2D4A3A;
    
    /* Ink colors — HIGH CONTRAST for readability */
    --ink-primary: #1A2B20;
    --ink-secondary: #3E5245;
    --ink-muted: #6B7D70;
    --ink-faint: #9AA89F;
    --ink-on-dark: #FAF7F0;
    
    /* Accents */
    --accent-heritage: #2D5F3F;
    --accent-copper: #B87333;
    --accent-sage: #8BA888;
    --accent-sand: #D4B896;
    --accent-clay: #C17B4E;
    
    /* Borders & Shadows */
    --border: #E5DFD0;
    --border-strong: #C9C0AB;
    --shadow-sm: 0 1px 3px rgba(26, 43, 32, 0.05);
    --shadow-md: 0 4px 12px rgba(26, 43, 32, 0.07);
    --shadow-lg: 0 12px 32px rgba(26, 43, 32, 0.10);
}

/* ============================================
   1. GLOBAL BASE
   ============================================ */
.stApp {
    background-color: var(--bg-paper);
    color: var(--ink-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 15px;
    line-height: 1.6;
}

.block-container {
    max-width: 1550px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Elegant scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg-paper); }
::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 10px;
    border: 2px solid var(--bg-paper);
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-sage); }

/* ============================================
   2. TYPOGRAPHY — HIGH READABILITY
   ============================================ */
h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink-primary) !important;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}

h1 { font-size: 2.4rem !important; line-height: 1.2 !important; }
h2 { font-size: 1.8rem !important; line-height: 1.3 !important; }
h3 { font-size: 1.4rem !important; line-height: 1.4 !important; }

p, span, div, label {
    color: var(--ink-primary);
    font-family: 'Inter', sans-serif;
}

strong, b {
    color: var(--ink-primary);
    font-weight: 700;
}

/* ============================================
   3. HERO STRIP
   ============================================ */
.hero-strip {
    background: linear-gradient(135deg, #2D4A3A 0%, #3E5F4D 50%, #4A6F58 100%);
    color: var(--ink-on-dark);
    padding: 3rem 3.5rem;
    border-radius: 28px;
    box-shadow: 0 20px 50px rgba(45, 74, 58, 0.20);
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}

.hero-strip::before {
    content: '🌿';
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 14rem;
    opacity: 0.08;
    filter: blur(3px);
}

.hero-strip::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent-copper), var(--accent-sand), transparent);
}

.hero-title {
    font-family: 'Fraunces', serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin-bottom: 0.8rem;
    position: relative;
    z-index: 2;
    letter-spacing: -0.03em;
}

.hero-accent {
    width: 80px;
    height: 3px;
    background: var(--accent-copper);
    border-radius: 2px;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 2;
}

.hero-subtitle {
    color: rgba(250, 247, 240, 0.92);
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 75%;
    margin-bottom: 1.8rem;
    position: relative;
    z-index: 2;
    font-weight: 400;
}

.hero-badges {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}

.hero-badge {
    background: rgba(250, 247, 240, 0.12);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(250, 247, 240, 0.25);
    color: #FFFFFF;
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 500;
    transition: all 0.3s ease;
}

.hero-badge:hover {
    background: rgba(250, 247, 240, 0.20);
    transform: translateY(-2px);
}

.hero-mini-stats {
    display: flex;
    gap: 3rem;
    margin-top: 2.2rem;
    padding-top: 1.8rem;
    border-top: 1px solid rgba(250, 247, 240, 0.20);
    position: relative;
    z-index: 2;
    flex-wrap: wrap;
}

.mini-stat-label {
    font-size: 0.72rem;
    color: rgba(250, 247, 240, 0.70);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.mini-stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'Fraunces', serif;
    letter-spacing: -0.02em;
}

/* ============================================
   4. SIDEBAR — PREMIUM HERITAGE DESIGN
   ============================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0EBE0 0%, #E8E0CF 100%) !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
}

/* Brand header */
.sidebar-brand {
    background: linear-gradient(135deg, var(--bg-sidebar-header) 0%, #3E5F4D 100%);
    color: var(--ink-on-dark);
    padding: 2rem 1.5rem 1.8rem;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
    overflow: hidden;
}

.sidebar-brand::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-copper), var(--accent-sand));
}

.sidebar-brand-icon {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    display: block;
}

.sidebar-brand-title {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.sidebar-brand-sub {
    font-size: 0.72rem;
    color: rgba(250, 247, 240, 0.75);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-weight: 600;
}

/* Sidebar blocks */
.sidebar-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.3rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.sidebar-block:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--border-strong);
}

.sidebar-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent-heritage);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}

.sidebar-title::before {
    content: '';
    width: 4px;
    height: 14px;
    background: var(--accent-copper);
    border-radius: 2px;
}

/* Sidebar navigation pills */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 4px !important;
    padding: 0.3rem !important;
    background: rgba(229, 223, 208, 0.4);
    border-radius: 14px;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.2s ease !important;
    color: var(--ink-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    border-left: 3px solid transparent !important;
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background-color: rgba(139, 168, 136, 0.12) !important;
    color: var(--accent-heritage) !important;
    border-left-color: var(--accent-sage) !important;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
    display: none;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
    background-color: var(--accent-heritage) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(45, 95, 63, 0.25) !important;
    border-left-color: var(--accent-copper) !important;
}

/* Commodity brief card */
.commodity-brief {
    background: linear-gradient(135deg, #F5F0E3 0%, #EFE7D3 100%);
    border-left: 4px solid var(--accent-heritage);
    padding: 1rem 1.1rem;
    border-radius: 0 14px 14px 0;
    margin-top: 0.8rem;
    transition: all 0.3s ease;
}

.commodity-brief:hover {
    transform: translateX(3px);
    box-shadow: var(--shadow-sm);
}

/* Sidebar decorative footer */
.sidebar-footer {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-top: 1rem;
    border-top: 1px solid var(--border);
    position: relative;
}

.sidebar-footer-icon {
    font-size: 1.8rem;
    margin-bottom: 0.4rem;
    opacity: 0.5;
}

.sidebar-footer-text {
    font-size: 0.68rem;
    color: var(--ink-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-weight: 700;
}

/* Sidebar form controls */
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stSlider > div {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.3rem 0.6rem;
}

section[data-testid="stSidebar"] label {
    color: var(--ink-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    margin-bottom: 0.4rem !important;
}

/* ============================================
   5. KPI CARDS
   ============================================ */
.intel-kpi {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
}

.intel-kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 3px;
    background: var(--accent-sage);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s ease;
}

.intel-kpi:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-lg);
    border-color: var(--border-strong);
}

.intel-kpi:hover::before {
    transform: scaleX(1);
}

.kpi-layer1 {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.kpi-layer2 {
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--ink-primary);
    line-height: 1.1;
    margin-bottom: 0.4rem;
    font-family: 'Fraunces', serif;
    letter-spacing: -0.02em;
}

.kpi-layer3 {
    font-size: 0.88rem;
    color: var(--ink-secondary);
    line-height: 1.5;
    font-weight: 500;
}

/* ============================================
   6. SECTION TITLES
   ============================================ */
.section-title {
    font-family: 'Fraunces', serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--accent-heritage) !important;
    border-bottom: 3px solid var(--accent-copper);
    padding-bottom: 0.6rem;
    margin-top: 2.8rem;
    margin-bottom: 1.5rem;
    display: inline-block;
    letter-spacing: -0.01em;
}

.section-subtitle {
    font-size: 0.98rem;
    color: var(--ink-secondary);
    margin-bottom: 1.8rem;
    font-style: italic;
    padding-left: 1rem;
    border-left: 3px solid var(--accent-sand);
    font-weight: 500;
}

/* ============================================
   7. CARDS (Insight, Watchlist, Rec, Priority)
   ============================================ */
.insight-card {
    background: linear-gradient(135deg, #F0F5EE 0%, #E8F0E5 100%);
    border-left: 4px solid var(--accent-heritage);
    padding: 1.4rem 1.7rem;
    border-radius: 4px 16px 16px 4px;
    color: var(--ink-primary);
    margin: 1.2rem 0;
    line-height: 1.75;
    transition: all 0.3s ease;
    font-size: 0.98rem;
    box-shadow: var(--shadow-sm);
}

.insight-card:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-md);
    border-left-width: 6px;
}

.watchlist-card {
    background: linear-gradient(135deg, #FDF5EC 0%, #FAECD9 100%);
    border-left: 4px solid var(--accent-copper);
    padding: 1.4rem 1.7rem;
    border-radius: 4px 16px 16px 4px;
    color: var(--ink-primary);
    margin: 1.2rem 0;
    line-height: 1.75;
    font-size: 0.98rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.watchlist-card:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-md);
    border-left-width: 6px;
}

.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-sand);
    padding: 1.3rem 1.6rem;
    border-radius: 4px 14px 14px 4px;
    color: var(--ink-primary);
    margin: 0.8rem 0;
    line-height: 1.7;
    font-size: 0.96rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.rec-card:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-md);
    border-left-width: 6px;
}

.priority-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 1.8rem 1.5rem;
    border-radius: 18px;
    color: var(--ink-primary);
    margin: 0.6rem 0;
    transition: all 0.4s ease;
    text-align: center;
    box-shadow: var(--shadow-sm);
}

.priority-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-lg);
    border-color: var(--accent-heritage);
}

/* ============================================
   8. TABS
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(229, 223, 208, 0.4);
    padding: 0.5rem;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 10px !important;
    color: var(--ink-secondary) !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.4rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
    font-size: 0.92rem !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(139, 168, 136, 0.15) !important;
    color: var(--accent-heritage) !important;
}

.stTabs [aria-selected="true"] {
    background-color: var(--bg-card) !important;
    color: var(--accent-heritage) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    font-weight: 700 !important;
}

/* ============================================
   9. BUTTONS & CONTROLS
   ============================================ */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.01em;
}

.stButton > button {
    background-color: var(--accent-heritage) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(45, 95, 63, 0.2) !important;
}

.stButton > button:hover {
    background-color: #223F30 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(45, 95, 63, 0.3) !important;
}

.stDownloadButton > button {
    background-color: var(--bg-card) !important;
    color: var(--accent-heritage) !important;
    border: 1.5px solid var(--accent-heritage) !important;
}

.stDownloadButton > button:hover {
    background-color: #F0F5EE !important;
    transform: translateY(-2px) !important;
}

/* Main area form labels */
.stSelectbox label, .stSlider label, .stRadio label,
.stMultiSelect label, .stNumberInput label {
    color: var(--ink-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em;
}

/* ============================================
   10. METRICS & DATAFRAMES
   ============================================ */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

div[data-testid="stMetric"] label {
    color: var(--ink-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--ink-primary) !important;
    font-family: 'Fraunces', serif !important;
    font-weight: 700 !important;
    font-size: 1.5rem;
}

header[data-testid="stHeader"] {
    background: rgba(250, 247, 240, 0.95) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
}

.stDataFrame {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

/* ============================================
   11. WARNINGS
   ============================================ */
.stWarning {
    border-radius: 12px !important;
    border-left: 4px solid var(--accent-copper) !important;
    background: #FDF5EC !important;
    color: var(--ink-primary) !important;
    padding: 1rem 1.2rem !important;
    font-size: 0.92rem;
}

/* ============================================
   12. ORGANIC DIVIDER
   ============================================ */
.organic-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-strong), var(--accent-sand), var(--border-strong), transparent);
    margin: 3rem 0;
}

/* ============================================
   13. CODE BLOCKS
   ============================================ */
.stCodeBlock {
    border-radius: 12px !important;
    border: 1px solid var(--border-strong) !important;
    background: #1A2B20 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET
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
# COMMODITY IDENTITY MAP — EARTHY REFINED PALETTE
# =========================================================
COMMODITY_IDENTITY = {
    "Kelapa Sawit": {"icon": "🌴", "color": "#3E5F4D", "color_light": "#6B9278", "sector": "Perkebunan Besar", "desc": "Primadona ekspor. Mendominasi Sumatera & Kalimantan."},
    "Kelapa": {"icon": "🥥", "color": "#5A7B8A", "color_light": "#89AAB8", "sector": "Perkebunan Rakyat", "desc": "Tanaman pesisir Nusantara. Potensi hilirisasi minyak & sabut."},
    "Karet": {"icon": "🌳", "color": "#8B6F4E", "color_light": "#B39471", "sector": "Perkebunan Campuran", "desc": "Sentra di Sumatera. Tantangan fluktuasi harga & peremajaan."},
    "Kopi": {"icon": "☕", "color": "#6B4E3D", "color_light": "#9C7B63", "sector": "Perkebunan Rakyat", "desc": "Ikon specialty coffee dunia (Gayo, Toraja, Kintamani)."},
    "Kakao": {"icon": "🍫", "color": "#7A5647", "color_light": "#A67B66", "sector": "Perkebunan Rakyat", "desc": "Sulawesi sebagai tulang punggung. Perlu peremajaan pohon."},
    "Teh": {"icon": "🍃", "color": "#5E7A54", "color_light": "#8FA885", "sector": "Perkebunan Besar", "desc": "Eksklusif dataran tinggi dingin (Jabar & Sumut)."},
    "Tebu": {"icon": "🌾", "color": "#A89558", "color_light": "#CBB879", "sector": "Perkebunan Strategis", "desc": "Bahan baku gula nasional (Jatim & Lampung)."}
}

SOFT_PALETTE = ["#6B9278", "#89AAB8", "#B39471", "#9C7B63", "#A67B66", "#8FA885", "#CBB879"]

def get_comm_attr(comm, attr="color"):
    return COMMODITY_IDENTITY.get(comm, {}).get(attr, "#6B9278")

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
        template="plotly_white",
        paper_bgcolor="#FAF7F0",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1A2B20", family="Inter, sans-serif", size=13),
        height=height,
        margin=dict(l=40, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor="#E5DFD0",
            zerolinecolor="#E5DFD0",
            tickfont=dict(size=11, color="#3E5245", family="Inter")
        ),
        yaxis=dict(
            gridcolor="#E5DFD0",
            zerolinecolor="#E5DFD0",
            tickfont=dict(size=11, color="#3E5245", family="Inter")
        ),
        title=dict(
            font=dict(size=17, color="#2D5F3F", family="Fraunces, serif", weight=600),
            x=0.02, xanchor="left"
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#2D5F3F",
            font=dict(color="#1A2B20", family="Inter", size=13)
        ),
        legend=dict(font=dict(color="#3E5245", size=12))
    )
    return fig

# =========================================================
# SIDEBAR — PREMIUM HERITAGE
# =========================================================
with st.sidebar:
    # Brand header
    st.markdown('''
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🌿</span>
        <div class="sidebar-brand-title">Plantation Intel</div>
        <div class="sidebar-brand-sub">Indonesia 2024</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Navigation block
    st.markdown('<div class="sidebar-block"><div class="sidebar-title">🧭 Navigasi</div>', unsafe_allow_html=True)
    menu = st.radio("Menu",
        ["🏠 Ringkasan Nasional", "🌴 Profil Komoditas", "🗺️ Profil Provinsi",
         "🔬 Eksplorasi Visual", "📊 Analisis Produksi", "🌍 Sebaran Wilayah",
         "📈 Proyeksi & Model", "🧠 Insight & Strategi", "📦 Data & Ekspor"],
        label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter block
    st.markdown('<div class="sidebar-block"><div class="sidebar-title">🎛️ Filter Data</div>', unsafe_allow_html=True)
    selected_commodity = st.selectbox("Komoditas Fokus", numeric_cols, index=0, key="side_comm")
    selected_province = st.selectbox("Wilayah Provinsi", ["Semua Provinsi"] + df["Provinsi"].tolist(), index=0, key="side_prov")
    top_n = st.slider("Top N Sentra", 5, 20, 10, key="side_topn")
    show_zeros = st.checkbox("Tampilkan wilayah tanpa produksi", value=True, key="side_zero")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Commodity brief
    comm_info = COMMODITY_IDENTITY[selected_commodity]
    st.markdown(f'''<div class="sidebar-block">
        <div class="sidebar-title">🌱 Commodity Brief</div>
        <div class="commodity-brief" style="border-left-color: {comm_info["color_light"]};">
            <div style="font-size:1.4rem; margin-bottom:0.4rem; color:#1A2B20; font-weight:700; font-family:'Fraunces',serif;">{comm_info["icon"]} {selected_commodity}</div>
            <div style="font-size:0.7rem; color:#6B7D70; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:0.5rem; font-weight:700;">{comm_info["sector"]}</div>
            <div style="font-size:0.88rem; color:#3E5245; line-height:1.6;">{comm_info["desc"]}</div>
        </div>
    </div>''', unsafe_allow_html=True)
    
    # Footer
    st.markdown('''
    <div class="sidebar-footer">
        <div class="sidebar-footer-icon">🌴</div>
        <div class="sidebar-footer-text">Tropical Heritage</div>
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
        <div class="hero-title">Plantation Intelligence</div>
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
    with c2: st.markdown(create_intel_kpi("Sentra Tertinggi", top_prov[:14], f"Output {format_ton(top_val)}", "🏆", "#B87333"), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Terdiversifikasi", diverse_prov[:14], f"{diverse_count} komoditas", "🌱", "#2D5F3F"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Konsentrasi Top-5", f"{top5_share:.1f}%", "Pangsa 5 provinsi", "🎯", "#8BA888"), unsafe_allow_html=True)

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
                      title="Kontribusi Output Perkebunan per Provinsi", color_discrete_sequence=["#B87333"])
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig2, 520), use_container_width=True, key="nat_bar_02")

    st.markdown('<div class="organic-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ Peta Struktur Komoditas Nasional</div>', unsafe_allow_html=True)
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        comp_df = active_df[numeric_cols].sum().reset_index()
        comp_df.columns = ["Komoditas", "Produksi"]
        fig_pie = px.pie(comp_df, values="Produksi", names="Komoditas", hole=0.65,
                         color="Komoditas", color_discrete_map={c: get_comm_attr(c, "color_light") for c in comp_df["Komoditas"]},
                         title="Komposisi Output Nasional")
        fig_pie.update_traces(textinfo='percent+label', textfont_size=12, textfont_color='#1A2B20')
        st.plotly_chart(apply_plantation_layout(fig_pie, 450), use_container_width=True, key="nat_pie_01")
    with col_t2:
        tm_df = active_df.copy()
        tm_df["Total"] = tm_df[numeric_cols].sum(axis=1)
        tm_df = tm_df[tm_df["Total"] > 0].nlargest(15, "Total")
        fig_tm = px.treemap(tm_df, path=["Provinsi"], values="Total", color="Total", color_continuous_scale=[[0, "#E5DFD0"], [0.5, "#8BA888"], [1, "#2D5F3F"]],
                            title="15 Wilayah Teratas (Treemap)")
        fig_tm.update_traces(textfont_color="#FFFFFF", textfont_size=13)
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
    <div style="background: #FFFFFF; border: 1px solid #E5DFD0; border-radius: 24px; padding: 2.8rem; margin-bottom: 2.2rem; box-shadow: 0 4px 24px rgba(26, 43, 32, 0.05); position: relative; overflow: hidden;">
        <div style="position:absolute; top:-20px; right:-20px; font-size:10rem; opacity:0.05; transform:rotate(15deg);">{info['icon']}</div>
        <div style="font-family:'Fraunces',serif; font-size:2.2rem; font-weight:700; color: #2D5F3F; margin-bottom: 0.8rem; letter-spacing: -0.02em;">{info['icon']} Profil Komoditas: {target}</div>
        <div style="font-size:1rem; color:#3E5245; line-height:1.75; font-weight:500;"><b style="color:#1A2B20;">Sektor:</b> {info['sector']}<br><b style="color:#1A2B20;">Potret:</b> {info['desc']}</div>
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
    with c2: st.markdown(create_intel_kpi("Sentra Utama", top_p[:14], "Kontributor terbesar", "🏆", "#B87333"), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Median Produksi", format_ton(med), "Tendensi sentral", "⚖️", "#8BA888"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Konsentrasi Top-5", f"{share5:.1f}%", "Pangsa 5 provinsi", "🎯", "#6B9278"), unsafe_allow_html=True)

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
                           color_continuous_scale=[[0, "#E5DFD0"], [0.5, info["color_light"]], [1, info["color"]]], title="Treemap Kontribusi Wilayah")
        fig_t.update_traces(textfont_color="#FFFFFF", textfont_size=13)
        st.plotly_chart(apply_plantation_layout(fig_t, 500), use_container_width=True, key=f"comm_tree_{target}")
        fig_p = px.pie(c_df.head(10), values=target, names="Provinsi", hole=0.65, title="Komposisi 10 Wilayah Teratas",
                       color_discrete_sequence=SOFT_PALETTE)
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
    <div style="background: #FFFFFF; border: 1px solid #E5DFD0; border-radius: 24px; padding: 2.8rem; margin-bottom: 2.2rem; box-shadow: 0 4px 24px rgba(26, 43, 32, 0.05); position: relative; overflow: hidden;">
        <div style="position:absolute; top:-30px; right:-10px; font-size:10rem; opacity:0.05; color: #2D5F3F;">🗺️</div>
        <div style="font-family:'Fraunces',serif; font-size:2.2rem; font-weight:700; color: #2D5F3F; margin-bottom: 0.8rem;">🗺️ Profil Perkebunan: {target_prov}</div>
        <div style="font-size:1rem; color:#3E5245; line-height:1.75; font-weight:500;">Total output <b style="color:#1A2B20;">{format_ton(tot_p)} ribu ton</b> ({nat_share:.2f}% nasional). Komoditas andalan: <b style="color:#1A2B20;">{dom_c}</b> ({dom_share:.1f}%).</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(create_intel_kpi("Total Output", format_num(tot_p), "Agregat komoditas", "📦", "#6B9278"), unsafe_allow_html=True)
    with c2: st.markdown(create_intel_kpi("Komoditas Andalan", dom_c, f"Penopang ({dom_share:.0f}%)", get_comm_attr(dom_c,"icon"), get_comm_attr(dom_c,"color_light")), unsafe_allow_html=True)
    with c3: st.markdown(create_intel_kpi("Sektor Aktif", str(active_c), f"dari {len(numeric_cols)}", "🌱", "#8BA888"), unsafe_allow_html=True)
    with c4: st.markdown(create_intel_kpi("Tingkat Dominasi", f"{dom_share:.0f}%", "Konsentrasi utama", "🎯", "#B87333"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌳 Struktur & Portofolio Komoditas</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        p_df = pd.DataFrame({"Komoditas": list(p_data.keys()), "Produksi": list(p_data.values())}).sort_values("Produksi", ascending=False)
        fig = px.bar(p_df, x="Komoditas", y="Produksi", text="Produksi", color="Komoditas",
                     color_discrete_map={c: get_comm_attr(c, "color_light") for c in p_df["Komoditas"]}, title="Portofolio Komoditas")
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(apply_plantation_layout(fig, 460), use_container_width=True, key=f"prov_bar_{target_prov}")
    with col2:
        fig2 = px.pie(p_df[p_df["Produksi"]>0], values="Produksi", names="Komoditas", hole=0.65, title="Komposisi Output Daerah",
                      color_discrete_sequence=SOFT_PALETTE)
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
                                        fill='toself', name=target_prov, line_color="#2D5F3F", fillcolor="rgba(45, 95, 63, 0.3)"))
        fig_r.add_trace(go.Scatterpolar(r=nat_norm + [nat_norm[0]], theta=numeric_cols + [numeric_cols[0]],
                                        fill='toself', name="Rata-rata Nasional", line_color="#B87333", fillcolor="rgba(184, 115, 51, 0.15)"))
        fig_r.update_layout(polar=dict(bgcolor="#FFFFFF",
                                       radialaxis=dict(visible=True, showticklabels=False, gridcolor="#E5DFD0"),
                                       angularaxis=dict(gridcolor="#E5DFD0", tickfont=dict(color="#3E5245", family="Inter"))),
                            showlegend=True, title="Radar (Normalisasi)")
        st.plotly_chart(apply_plantation_layout(fig_r, 500), use_container_width=True, key=f"prov_radar_{target_prov}")
    with col_r2:
        bench_df = pd.DataFrame({"Komoditas": numeric_cols, "Provinsi": p_values, "Nasional": list(nat_avg.values())})
        bench_df["Deviasi"] = bench_df["Provinsi"] - bench_df["Nasional"]
        colors = ["#6B9278" if v >= 0 else "#B39471" for v in bench_df["Deviasi"]]
        fig3 = go.Figure(go.Bar(y=bench_df["Komoditas"], x=bench_df["Deviasi"], orientation='h', marker_color=colors))
        fig3.update_layout(title="Deviasi vs Rata-rata Nasional", xaxis_title="Deviasi (Ribu Ton)")
        st.plotly_chart(apply_plantation_layout(fig3, 500), use_container_width=True, key=f"prov_bench_{target_prov}")

    dep_note = "⚠️ Sangat bergantung pada satu komoditas." if dom_share > 70 else "✅ Portofolio relatif seimbang."
    st.markdown(f'<div class="insight-card">🗺️ <b>{target_prov}</b>: <b>{active_c}</b> komoditas aktif. {dep_note} Strategi: perkuat klaster unggulan & hilirisasi lokal.</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 4: EKSPLORASI VISUAL
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
            fig2 = px.bar(top_total, x="Total", y="Provinsi", orientation='h', text="Total", title="Top Provinsi Total Produksi", color_discrete_sequence=["#B87333"])
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
            fig = px.scatter(active_df, x=x_v, y=y_v, hover_name="Provinsi", title=f"{x_v} vs {y_v}", color_discrete_sequence=["#2D5F3F"])
            st.plotly_chart(apply_plantation_layout(fig, 550), use_container_width=True, key="eda_rel_scatter")
            corr = active_df[[x_v, y_v]].corr().iloc[0,1]
            strength = "kuat" if abs(corr)>0.6 else "sedang" if abs(corr)>0.3 else "lemah"
            st.markdown(f'<div class="insight-card">🔗 <b>Korelasi:</b> {corr:.3f} ({strength})</div>', unsafe_allow_html=True)
        else: st.warning("Pilih dua komoditas berbeda.")
    with tab4:
        corr_m = active_df[numeric_cols].corr()
        fig = px.imshow(corr_m, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale=[[0,"#A67B66"],[0.5,"#FAF7F0"],[1,"#2D5F3F"]],
                        title="Matriks Korelasi Agro-Ekologis")
        fig.update_traces(textfont=dict(color="#1A2B20", size=12, family="Inter"), xgap=3, ygap=3)
        fig.update_xaxes(side="bottom", tickangle=30); fig.update_yaxes(autorange="reversed")
        fig.update_layout(paper_bgcolor="#FAF7F0", plot_bgcolor="#FFFFFF",
                          font=dict(color="#1A2B20"), margin=dict(l=40,r=30,t=60,b=40))
        fig.update_coloraxes(colorbar=dict(title=dict(text="Korelasi", font=dict(color="#3E5245")),
                                            tickfont=dict(color="#3E5245"), len=0.75))
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
                st.markdown(create_intel_kpi("Total Produksi", format_num(total_p), "Ribu Ton", "📦", "#6B9278"), unsafe_allow_html=True)
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
        fig.add_trace(go.Bar(name=p1, x=comp["Komoditas"], y=comp[p1], marker_color="#6B9278"))
        fig.add_trace(go.Bar(name=p2, x=comp["Komoditas"], y=comp[p2], marker_color="#B87333"))
        fig.update_layout(barmode='group', title=f"{p1} vs {p2}")
        st.plotly_chart(apply_plantation_layout(fig, 480), use_container_width=True, key="duel_chart")
        w1, w2 = (comp[p1] >comp[p2]).sum(), (comp[p2] >comp[p1]).sum()
        st.markdown(f'<div class="insight-card">⚔️ {p1} unggul <b>{w1}</b> komoditas | {p2} unggul <b>{w2}</b> komoditas.</div>', unsafe_allow_html=True)
    with tab2:
        c1, c2 = st.columns(2)
        x_v = c1.selectbox("X", numeric_cols, index=0, key="rel_x")
        y_v = c2.selectbox("Y", numeric_cols, index=1, key="rel_y")
        if x_v != y_v:
            fig = px.scatter(active_df, x=x_v, y=y_v, hover_name="Provinsi", title=f"{x_v} vs {y_v}", color_discrete_sequence=["#2D5F3F"])
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
        fig.add_hline(y=1, line_dash="dash", line_color="#B87333")
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
                      color_continuous_scale=[[0, "#E5DFD0"], [1, get_comm_attr(selected_commodity, "color_light")]],
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
        k1.markdown(create_intel_kpi("Sentra Nasional", str(kc.get("Nasional",0)), "Anchor production", "🏆", "#B87333"), unsafe_allow_html=True)
        k2.markdown(create_intel_kpi("Sentra Regional", str(kc.get("Regional",0)), "Pillar produksi", "🌾", "#6B9278"), unsafe_allow_html=True)
        k3.markdown(create_intel_kpi("Penyangga", str(kc.get("Penyangga",0)), "Stabilisator", "🌱", "#8BA888"), unsafe_allow_html=True)
        k4.markdown(create_intel_kpi("Minor", str(kc.get("Minor",0)), "Niche", "🍃", "#B39471"), unsafe_allow_html=True)

# =========================================================
# PAGE 7: PROYEKSI & MODEL
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
                    fig.add_trace(go.Scatter(x=mdf[x_var], y=mdf[y_var], mode="markers", name="Aktual", marker=dict(color="#2D5F3F", size=9, symbol="circle")))
                    sort_idx = np.argsort(mdf[x_var].values)
                    fig.add_trace(go.Scatter(x=mdf[x_var].values[sort_idx], y=pred[sort_idx], mode="lines", name="Regresi", line=dict(color="#B87333", width=3)))
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
            fig_fc.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc[fc_comm], name="Panen 2024", marker_color="#B39471"))
            fig_fc.add_trace(go.Bar(x=top_fc["Provinsi"], y=top_fc["Proyeksi_2025"], name="Proyeksi 2025", marker_color="#6B9278"))
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
                    fig_imp = px.bar(x=imp.values, y=imp.index, orientation='h', title="Feature Importance", color_discrete_sequence=["#6B9278"])
                    st.plotly_chart(apply_plantation_layout(fig_imp, 450), use_container_width=True, key="rf_imp")
                with c2:
                    comp_df = pd.DataFrame({"Aktual": y_te, "Prediksi": y_pred})
                    fig_sc = px.scatter(comp_df, x="Aktual", y="Prediksi", title="Aktual vs Prediksi (Test Set)", color_discrete_sequence=["#B87333"])
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
                                    color_discrete_sequence=["#8BA888"])
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
    with p1: st.markdown('<div class="priority-card"><b style="font-family:Fraunces,serif; font-size:1.2rem; color:#2D5F3F;">🏭 Hilirisasi & Nilai Tambah</b><br><span style="color:#3E5245; font-size:0.92rem;">Pengolahan CPO, karet olahan, kopi specialty.</span></div>', unsafe_allow_html=True)
    with p2: st.markdown('<div class="priority-card"><b style="font-family:Fraunces,serif; font-size:1.2rem; color:#2D5F3F;">🌱 Diversifikasi & Ketahanan</b><br><span style="color:#3E5245; font-size:0.92rem;">Tanaman sela & agroforestri berkelanjutan.</span></div>', unsafe_allow_html=True)
    with p3: st.markdown('<div class="priority-card"><b style="font-family:Fraunces,serif; font-size:1.2rem; color:#2D5F3F;">📈 Digitalisasi & Data</b><br><span style="color:#3E5245; font-size:0.92rem;">Precision agriculture & dashboard real-time.</span></div>', unsafe_allow_html=True)

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
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; padding:3rem 2rem; margin-top:3rem; background: #FFFFFF; border-radius: 24px; border: 1px solid #E5DFD0; box-shadow: 0 -4px 24px rgba(26, 43, 32, 0.04);">
    <div style="font-size:2.5rem; margin-bottom:0.5rem; opacity:0.7;">🌿</div>
    <h3 style="font-family:'Fraunces',serif; font-size:1.6rem; font-weight:700; color: #2D5F3F; margin-bottom:0.6rem; letter-spacing:-0.02em;">Plantation Intelligence Dashboard</h3>
    <p style="color:#3E5245; font-size:0.95rem; margin-bottom:0.4rem; font-weight:500;">UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar</p>
    <p style="color:#6B7D70; font-size:0.88rem; margin-top:0.8rem;">Streamlit + Plotly + Scikit-Learn untuk perencanaan strategis sektor perkebunan Indonesia</p>
    <div style="width:60px; height:2px; background: linear-gradient(90deg, #2D5F3F, #B87333); margin: 1.5rem auto; border-radius:2px;"></div>
    <p style="color:#9AA89F; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:600;">© 2026 | Sumber: BPS — Produksi Tanaman Perkebunan Menurut Provinsi, 2024</p>
</div>
""", unsafe_allow_html=True)
