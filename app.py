import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.tree import export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import meanabsoluteerror, meansquarederror, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.modelselection import traintest_split
import warnings
warnings.filterwarnings('ignore')

=========================================================
PAGE CONFIG
=========================================================
st.setpageconfig(
    page_title="Plantation Intelligence Dashboard",
    page_icon="🌿",
    layout="wide",
    initialsidebarstate="expanded"
)

=========================================================
#🗺️ GEOJSON & PROVINCE MAPPING
=========================================================
PROVINCE_MAPPING = {
    "DKI JAKARTA": "Jakarta Raya",
    "DI YOGYAKARTA": "Yogyakarta",
    "KEP. BANGKA BELITUNG": "Kepulauan Bangka Belitung",
    "KEP. RIAU": "Kepulauan Riau",
    "NUSA TENGGARA BARAT": "Nusa Tenggara Barat",
    "NUSA TENGGARA TIMUR": "Nusa Tenggara Timur",
    "SULAWESI TENGGARA": "Sulawesi Tenggara",
    "KALIMANTAN BARAT": "Kalimantan Barat",
    "KALIMANTAN TENGAH": "Kalimantan Tengah",
    "KALIMANTAN SELATAN": "Kalimantan Selatan",
    "KALIMANTAN TIMUR": "Kalimantan Timur",
    "KALIMANTAN UTARA": "Kalimantan Utara",
    "SUMATERA UTARA": "Sumatera Utara",
    "SUMATERA BARAT": "Sumatera Barat",
    "SUMATERA SELATAN": "Sumatera Selatan",
    "PAPUA BARAT DAYA": "Papua Barat",
    "PAPUA PEGUNUNGAN": "Papua",
    "PAPUA SELATAN": "Papua",
    "PAPUA TENGAH": "Papua",
}

@st.cachedata(showspinner="Memuat peta Indonesia...")
def loadindonesiageojson():
    urls = [
        "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province.json",
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            continue
    return None

def normalizeprovincename(name):
    return PROVINCE_MAPPING.get(name, name)

def createchoroplethmap(dfmap, valuecol, title, colorlabel="Produksi (Ribu Ton)", isaggregate=False):
    geojson = loadindonesiageojson()
    if geojson is None:
        return None
    
    mapdf = dfmap.copy()
    mapdf["ProvinsiNormalized"] = mapdf["Provinsi"].apply(normalizeprovince_name)
    
    if is_aggregate:
        colorscale = [
            [0, "#FDF5EC"], [0.25, "#E8C9A0"], [0.5, "#C28F6A"],
            [0.75, "#8B5A3C"], [1, "#4A2C1A"]
        ]
    else:
        colorscale = [
            [0, "#F5F0E3"], [0.25, "#D4B896"], [0.5, "#8BA888"],
            [0.75, "#2D5F3F"], [1, "#1A2B20"]
        ]
    
    fig = px.choropleth(
        map_df, geojson=geojson, featureidkey="properties.Propinsi",
        locations="ProvinsiNormalized", color=valuecol,
        colorcontinuousscale=colorscale, scope="asia",
        labels={valuecol: colorlabel}, title=title,
        hovername="Provinsi", hoverdata={"ProvinsiNormalized": False, valuecol: ":,.2f"}
    )
    
    fig.update_geos(
        showcountries=True, countrycolor="#C9C0AB",
        showcoastlines=True, coastlinecolor="#8BA888",
        showland=True, landcolor="#F5F0E3",
        showocean=True, oceancolor="#E8F0E5",
        showlakes=False, projection_type="mercator",
        fitbounds="locations", lataxisrange=[-12, 8], lonaxisrange=[94, 142]
    )
    
    fig.update_layout(
        paperbgcolor="#FAF7F0", plotbgcolor="#FFFFFF",
        font=dict(color="#1A2B20", family="Inter, sans-serif"),
        margin={"r": 0, "t": 60, "l": 0, "b": 40}, height=600,
        title=dict(font=dict(size=17, color="#2D5F3F", family="Fraunces, serif", weight=600), x=0.02, xanchor="left"),
        coloraxis_colorbar=dict(
            title=dict(text=color_label, font=dict(color="#3E5245")),
            tickfont=dict(color="#3E5245"), len=0.8, x=1.02,
            thickness=20, outlinecolor="#C9C0AB", outlinewidth=1
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#2D5F3F", font=dict(color="#1A2B20", family="Inter", size=13))
    )
    return fig

=========================================================
🎨 ULTRA PREMIUM BOTANICAL THEME
=========================================================
st.markdown("""

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800;9..144,900&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-paper: #FAF7F0;
    --bg-paper-deep: #F3EEE2;
    --bg-card: #FFFFFF;
    --bg-sidebar: #F0EBE0;
    --bg-sidebar-header: #1E3A2B;
    --ink-primary: #1A2B20;
    --ink-secondary: #3E5245;
    --ink-muted: #6B7D70;
    --ink-faint: #9AA89F;
    --ink-on-dark: #FAF7F0;
    --accent-heritage: #2D5F3F;
    --accent-heritage-deep: #1E3A2B;
    --accent-copper: #B87333;
    --accent-copper-light: #D4A574;
    --accent-sage: #8BA888;
    --accent-sage-deep: #6B8E76;
    --accent-sand: #D4B896;
    --accent-clay: #C17B4E;
    --accent-cream: #F5EFE0;
    --border: #E5DFD0;
    --border-strong: #C9C0AB;
    --shadow-xs: 0 1px 2px rgba(26, 43, 32, 0.03);
    --shadow-sm: 0 2px 8px rgba(26, 43, 32, 0.04);
    --shadow-md: 0 8px 24px rgba(26, 43, 32, 0.06);
    --shadow-lg: 0 16px 48px rgba(26, 43, 32, 0.08);
    --shadow-xl: 0 24px 64px rgba(26, 43, 32, 0.12);
    --shadow-glow-green: 0 0 40px rgba(139, 168, 136, 0.25);
    --shadow-glow-copper: 0 0 40px rgba(184, 115, 51, 0.20);
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;
}

/* ============================================
   NOISE TEXTURE OVERLAY (Organic Feel)
   ============================================ */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.04 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1;
    opacity: 0.6;
}

.stApp {
    background-color: var(--bg-paper);
    background-image: 
        radial-gradient(at 20% 10%, rgba(139, 168, 136, 0.08) 0px, transparent 50%),
        radial-gradient(at 80% 90%, rgba(212, 184, 150, 0.08) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(184, 115, 51, 0.03) 0px, transparent 60%);
    color: var(--ink-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    font-feature-settings: "cv11", "ss01", "ss03";
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.block-container { 
    max-width: 1550px; 
    padding-top: 2.5rem; 
    padding-bottom: 3rem;
    position: relative;
    z-index: 2;
}

/ Scrollbar premium /
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, var(--accent-sage), var(--accent-heritage));
    border-radius: 10px; 
    border: 2px solid var(--bg-paper); 
}
::-webkit-scrollbar-thumb:hover { 
    background: linear-gradient(180deg, var(--accent-copper), var(--accent-clay));
}

/ Scroll progress indicator /
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    height: 3px;
    width: 0%;
    background: linear-gradient(90deg, var(--accent-heritage), var(--accent-copper), var(--accent-sand));
    z-index: 9999;
    animation: scrollProgress 2s ease-out forwards;
    box-shadow: 0 0 10px rgba(45, 95, 63, 0.4);
}

@keyframes scrollProgress {
    0% { width: 0%; }
    100% { width: 100%; }
}

/* ============================================
   TYPOGRAPHY — EDITORIAL PREMIUM
   ============================================ */
h1, h2, h3, h4 { 
    font-family: 'Fraunces', Georgia, serif !important; 
    color: var(--ink-primary) !important; 
    letter-spacing: -0.025em; 
    font-weight: 700 !important;
    font-variation-settings: "opsz" 144, "SOFT" 50;
}
h1 { font-size: 2.6rem !important; line-height: 1.1 !important; } 
h2 { font-size: 1.9rem !important; line-height: 1.2 !important; } 
h3 { font-size: 1.5rem !important; line-height: 1.3 !important; }
h4 { font-size: 1.2rem !important; line-height: 1.4 !important; }

p, span, div, label {
    color: var(--ink-primary);
    font-family: 'Inter', sans-serif;
}

strong, b {
    color: var(--ink-primary);
    font-weight: 700;
    letter-spacing: -0.01em;
}

/* ============================================
   ANIMATIONS — CINEMATIC & ORGANIC
   ============================================ */
@keyframes fadeInUp { 
    from { opacity: 0; transform: translateY(30px); } 
    to { opacity: 1; transform: translateY(0); } 
}
@keyframes fadeInScale { 
    from { opacity: 0; transform: scale(0.94); } 
    to { opacity: 1; transform: scale(1); } 
}
@keyframes slideInLeft { 
    from { opacity: 0; transform: translateX(-40px); } 
    to { opacity: 1; transform: translateX(0); } 
}
@keyframes slideInRight { 
    from { opacity: 0; transform: translateX(40px); } 
    to { opacity: 1; transform: translateX(0); } 
}
@keyframes gentleSway { 
    0%, 100% { transform: rotate(-3deg) translateY(0); } 
    50% { transform: rotate(3deg) translateY(-5px); } 
}
@keyframes shimmer { 
    0% { background-position: -200% center; } 
    100% { background-position: 200% center; } 
}
@keyframes breathe { 
    0%, 100% { transform: scale(1); opacity: 0.9; } 
    50% { transform: scale(1.02); opacity: 1; } 
}
@keyframes gradientShift { 
    0% { background-position: 0% 50%; } 
    50% { background-position: 100% 50%; } 
    100% { background-position: 0% 50%; } 
}
@keyframes pulse { 
    0%, 100% { opacity: 1; transform: scale(1); } 
    50% { opacity: 0.85; transform: scale(1.05); } 
}
@keyframes float {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-15px) rotate(3deg); }
}
@keyframes auroraGlow {
    0%, 100% { 
        background-position: 0% 50%;
        filter: hue-rotate(0deg);
    }
    50% { 
        background-position: 100% 50%;
        filter: hue-rotate(20deg);
    }
}
@keyframes borderFlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes leafFall {
    0% { 
        transform: translate(0, -20px) rotate(0deg);
        opacity: 0;
    }
    10% { opacity: 0.5; }
    90% { opacity: 0.3; }
    100% { 
        transform: translate(50px, 100vh) rotate(720deg);
        opacity: 0;
    }
}
@keyframes textReveal {
    from { 
        clip-path: inset(0 100% 0 0);
        opacity: 0;
    }
    to { 
        clip-path: inset(0 0 0 0);
        opacity: 1;
    }
}
@keyframes counterUp {
    from { 
        opacity: 0;
        transform: translateY(20px);
    }
    to { 
        opacity: 1;
        transform: translateY(0);
    }
}

/* ============================================
   HERO STRIP — AURORA BOTANICAL
   ============================================ */
.hero-strip {
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(139, 168, 136, 0.3) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 30%, rgba(212, 184, 150, 0.2) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(184, 115, 51, 0.15) 0%, transparent 60%),
        linear-gradient(135deg, #1E3A2B 0%, #2D5F3F 40%, #3E7550 100%);
    background-size: 200% 200%, 200% 200%, 200% 200%, 100% 100%;
    animation: gradientShift 20s ease infinite, fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1);
    color: var(--ink-on-dark);
    padding: 3.5rem 4rem;
    border-radius: var(--radius-xl);
    box-shadow: 
        0 30px 60px rgba(30, 58, 43, 0.25),
        0 0 0 1px rgba(255, 255, 255, 0.05) inset,
        0 2px 0 rgba(255, 255, 255, 0.08) inset;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.hero-strip::before { 
    content: '🌿'; 
    position: absolute; 
    right: 4rem; 
    top: 50%; 
    transform: translateY(-50%); 
    font-size: 16rem; 
    opacity: 0.07; 
    filter: blur(4px) saturate(0.5);
    animation: gentleSway 8s ease-in-out infinite;
}

.hero-strip::after { 
    content: ''; 
    position: absolute; 
    top: 0; left: 0; right: 0; 
    height: 1px; 
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    background-size: 200% 100%;
    animation: shimmer 4s linear infinite;
}

.hero-title { 
    font-family: 'Fraunces', serif !important; 
    font-size: 3.2rem !important; 
    font-weight: 800 !important; 
    color: #FFFFFF !important; 
    margin-bottom: 1rem; 
    position: relative; 
    z-index: 2; 
    letter-spacing: -0.035em;
    line-height: 1.05 !important;
    animation: textReveal 1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s backwards;
    font-variation-settings: "opsz" 144, "SOFT" 0;
}

.hero-title::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 0;
    width: 80px;
    height: 4px;
    background: linear-gradient(90deg, var(--accent-copper), var(--accent-sand), transparent);
    border-radius: 2px;
    animation: slideInLeft 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.5s backwards;
}

.hero-accent { display: none; }

.hero-subtitle { 
    color: rgba(250, 247, 240, 0.88); 
    font-size: 1.1rem; 
    line-height: 1.7; 
    max-width: 70%; 
    margin-bottom: 2rem; 
    position: relative; 
    z-index: 2; 
    font-weight: 400;
    letter-spacing: -0.005em;
    animation: fadeInUp 0.8s ease-out 0.6s backwards;
}

.hero-badges { 
    display: flex; 
    gap: 0.75rem; 
    flex-wrap: wrap; 
    position: relative; 
    z-index: 2; 
    animation: fadeInUp 0.8s ease-out 0.8s backwards; 
}

.hero-badge { 
    background: rgba(250, 247, 240, 0.08); 
    backdrop-filter: blur(12px) saturate(150%);
    -webkit-backdrop-filter: blur(12px) saturate(150%);
    border: 1px solid rgba(250, 247, 240, 0.15); 
    color: #FFFFFF; 
    padding: 0.55rem 1.3rem; 
    border-radius: 999px; 
    font-size: 0.85rem; 
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
    overflow: hidden;
}

.hero-badge::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.6s ease;
}

.hero-badge:hover::before { left: 100%; }
.hero-badge:nth-child(1) { animation-delay: 0.9s; } 
.hero-badge:nth-child(2) { animation-delay: 1.0s; } 
.hero-badge:nth-child(3) { animation-delay: 1.1s; } 
.hero-badge:nth-child(4) { animation-delay: 1.2s; }

.hero-badge:hover { 
    background: rgba(250, 247, 240, 0.18); 
    transform: translateY(-4px); 
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    border-color: rgba(250, 247, 240, 0.3);
}

.hero-mini-stats { 
    display: flex; 
    gap: 3.5rem; 
    margin-top: 2.5rem; 
    padding-top: 2rem; 
    border-top: 1px solid rgba(250, 247, 240, 0.12); 
    position: relative; 
    z-index: 2; 
    flex-wrap: wrap; 
    animation: fadeInUp 0.8s ease-out 1.2s backwards; 
}

.mini-stat {
    position: relative;
    padding-right: 3.5rem;
}

.mini-stat:not(:last-child)::after {
    content: '';
    position: absolute;
    right: 0; top: 50%;
    transform: translateY(-50%);
    width: 1px;
    height: 32px;
    background: rgba(250, 247, 240, 0.15);
}

.mini-stat-label { 
    font-size: 0.68rem; 
    color: rgba(250, 247, 240, 0.6); 
    text-transform: uppercase; 
    letter-spacing: 0.18em; 
    font-weight: 600; 
    margin-bottom: 0.5rem; 
}

.mini-stat-value { 
    font-size: 1.8rem; 
    font-weight: 700; 
    color: #FFFFFF; 
    font-family: 'Fraunces', serif; 
    letter-spacing: -0.025em;
    font-variation-settings: "opsz" 144;
    text-shadow: 0 2px 20px rgba(0, 0, 0, 0.15);
}

/* ============================================
   SIDEBAR — BOTANICAL ATELIER
   ============================================ */
section[data-testid="stSidebar"] { 
    background: 
        linear-gradient(180deg, #F0EBE0 0%, #E8E0CF 100%) !important; 
    padding: 0 !important; 
    animation: slideInLeft 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 1px; height: 100%;
    background: linear-gradient(180deg, transparent, var(--border-strong), transparent);
    z-index: 10;
}

section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.sidebar-brand { 
    background: 
        radial-gradient(ellipse at 30% 20%, rgba(139, 168, 136, 0.2) 0%, transparent 60%),
        linear-gradient(135deg, var(--bg-sidebar-header) 0%, #2D5F3F 100%);
    color: var(--ink-on-dark); 
    padding: 2.5rem 1.8rem 2.2rem; 
    margin: -1rem -1rem 1.8rem -1rem; 
    position: relative; 
    overflow: hidden; 
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}

.sidebar-brand::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(212, 184, 150, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: float 10s ease-in-out infinite;
}

.sidebar-brand::after { 
    content: ''; 
    position: absolute; 
    bottom: 0; left: 0; right: 0; 
    height: 1px; 
    background: linear-gradient(90deg, transparent, var(--accent-copper), var(--accent-sand), transparent);
    background-size: 200% 100%;
    animation: shimmer 5s linear infinite;
}

.sidebar-brand-icon { 
    font-size: 2.8rem; 
    margin-bottom: 0.8rem; 
    display: block; 
    animation: gentleSway 6s ease-in-out infinite;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));
}

.sidebar-brand-title { 
    font-family: 'Fraunces', serif; 
    font-size: 1.6rem; 
    font-weight: 700; 
    color: #FFFFFF; 
    margin-bottom: 0.3rem; 
    letter-spacing: -0.025em;
    font-variation-settings: "opsz" 144;
}

.sidebar-brand-sub { 
    font-size: 0.68rem; 
    color: rgba(250, 247, 240, 0.6); 
    text-transform: uppercase; 
    letter-spacing: 0.25em; 
    font-weight: 600;
    position: relative;
    padding-left: 20px;
}

.sidebar-brand-sub::before {
    content: '';
    position: absolute;
    left: 0; top: 50%;
    transform: translateY(-50%);
    width: 12px; height: 1px;
    background: var(--accent-copper);
}

.sidebar-block { 
    background: var(--bg-card); 
    border: 1px solid var(--border); 
    border-radius: var(--radius-lg); 
    padding: 1.5rem; 
    margin-bottom: 1.3rem; 
    box-shadow: var(--shadow-sm); 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
    overflow: hidden;
}

.sidebar-block::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
}

.sidebar-block:nth-child(2) { animation-delay: 0.1s; } 
.sidebar-block:nth-child(3) { animation-delay: 0.2s; } 
.sidebar-block:nth-child(4) { animation-delay: 0.3s; }

.sidebar-block:hover { 
    box-shadow: var(--shadow-md); 
    border-color: var(--border-strong); 
    transform: translateY(-3px);
}

.sidebar-title { 
    font-size: 0.68rem; 
    font-weight: 700; 
    color: var(--accent-heritage); 
    text-transform: uppercase; 
    letter-spacing: 0.18em; 
    margin-bottom: 1rem; 
    display: flex; 
    align-items: center; 
    gap: 0.6rem; 
    padding-bottom: 0.7rem; 
    border-bottom: 1px solid var(--border);
    font-family: 'Inter', sans-serif;
}

.sidebar-title::before { 
    content: ''; 
    width: 3px; 
    height: 14px; 
    background: linear-gradient(180deg, var(--accent-copper), var(--accent-sand));
    border-radius: 2px; 
    animation: pulse 3s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(184, 115, 51, 0.3);
}

/ Radio nav pills /
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] { 
    gap: 4px !important; 
    padding: 0.4rem !important; 
    background: rgba(229, 223, 208, 0.5);
    border-radius: var(--radius-md); 
    backdrop-filter: blur(10px);
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label { 
    border-radius: var(--radius-sm) !important; 
    padding: 0.75rem 1.1rem !important; 
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important; 
    color: var(--ink-secondary) !important; 
    font-weight: 500 !important; 
    font-size: 0.88rem !important; 
    border-left: 3px solid transparent !important; 
    margin-bottom: 0 !important;
    position: relative;
    overflow: hidden;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 0;
    background: linear-gradient(90deg, rgba(139, 168, 136, 0.1), transparent);
    transition: width 0.4s ease;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover::before {
    width: 100%;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover { 
    color: var(--accent-heritage) !important; 
    border-left-color: var(--accent-sage) !important; 
    transform: translateX(4px) !important; 
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child { display: none; }

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[aria-checked="true"] { 
    background: linear-gradient(135deg, var(--accent-heritage) 0%, #3E7550 100%) !important; 
    color: #FFFFFF !important; 
    font-weight: 600 !important; 
    box-shadow: 
        0 4px 16px rgba(45, 95, 63, 0.3),
        0 2px 4px rgba(45, 95, 63, 0.2) !important; 
    border-left-color: var(--accent-copper) !important;
}

.commodity-brief { 
    background: linear-gradient(135deg, #F5EFE0 0%, #EDE4D0 100%);
    border-left: 4px solid var(--accent-heritage); 
    padding: 1.2rem 1.3rem; 
    border-radius: 0 var(--radius-md) var(--radius-md) 0; 
    margin-top: 0.9rem; 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}

.commodity-brief::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(139, 168, 136, 0.15) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(30%, -30%);
}

.commodity-brief:hover { 
    transform: translateX(6px); 
    box-shadow: var(--shadow-md);
}

.sidebar-footer { 
    text-align: center; 
    padding: 2rem 1rem; 
    margin-top: 1.5rem; 
    border-top: 1px solid var(--border); 
    position: relative; 
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.6s backwards; 
}

.sidebar-footer-icon { 
    font-size: 2.2rem; 
    margin-bottom: 0.5rem; 
    opacity: 0.6; 
    animation: gentleSway 7s ease-in-out infinite;
    display: inline-block;
}

.sidebar-footer-text { 
    font-size: 0.65rem; 
    color: var(--ink-muted); 
    letter-spacing: 0.25em; 
    text-transform: uppercase; 
    font-weight: 700;
}

/ Form controls /
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stSlider > div {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.3rem 0.6rem;
    transition: all 0.3s ease;
}

section[data-testid="stSidebar"] .stSelectbox > div:focus-within,
section[data-testid="stSidebar"] .stSlider > div:focus-within {
    border-color: var(--accent-sage);
    box-shadow: 0 0 0 3px rgba(139, 168, 136, 0.1);
}

section[data-testid="stSidebar"] label {
    color: var(--ink-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    margin-bottom: 0.4rem !important;
    letter-spacing: 0.01em;
}

/* ============================================
   INTEL KPI CARDS — GLASS PREMIUM
   ============================================ */
.intel-kpi { 
    background: 
        linear-gradient(145deg, rgba(255,255,255,1) 0%, rgba(250,247,240,0.7) 100%);
    border: 1px solid var(--border); 
    border-radius: var(--radius-lg); 
    padding: 1.7rem 1.6rem; 
    box-shadow: var(--shadow-sm); 
    transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1); 
    height: 100%; 
    position: relative; 
    overflow: hidden; 
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    backdrop-filter: blur(10px);
}

.intel-kpi:nth-child(1) { animation-delay: 0.1s; } 
.intel-kpi:nth-child(2) { animation-delay: 0.2s; } 
.intel-kpi:nth-child(3) { animation-delay: 0.3s; } 
.intel-kpi:nth-child(4) { animation-delay: 0.4s; }

.intel-kpi::before { 
    content: ''; 
    position: absolute; 
    top: 0; left: 0; 
    width: 100%; 
    height: 3px; 
    background: linear-gradient(90deg, var(--accent-sage), var(--accent-copper));
    transform: scaleX(0); 
    transform-origin: left; 
    transition: transform 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.intel-kpi::after {
    content: '';
    position: absolute;
    top: -50%; right: -30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139, 168, 136, 0.08) 0%, transparent 70%);
    border-radius: 50%;
    opacity: 0;
    transition: opacity 0.5s ease;
}

.intel-kpi:hover { 
    transform: translateY(-10px) scale(1.02); 
    box-shadow: 
        var(--shadow-lg),
        0 0 0 1px var(--border-strong);
    border-color: var(--accent-sage);
}

.intel-kpi:hover::before { transform: scaleX(1); }
.intel-kpi:hover::after { opacity: 1; }

.kpi-layer1 { 
    font-size: 0.7rem; 
    font-weight: 700; 
    color: var(--ink-muted); 
    text-transform: uppercase; 
    letter-spacing: 0.12em; 
    margin-bottom: 0.8rem; 
    display: flex; 
    align-items: center; 
    gap: 0.5rem;
    font-family: 'Inter', sans-serif;
}

.kpi-layer2 { 
    font-size: 2.3rem; 
    font-weight: 700; 
    color: var(--ink-primary); 
    line-height: 1.05; 
    margin-bottom: 0.5rem; 
    font-family: 'Fraunces', serif; 
    letter-spacing: -0.03em;
    font-variation-settings: "opsz" 144, "SOFT" 50;
    transition: all 0.4s ease;
    animation: counterUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.3s backwards;
}

.intel-kpi:hover .kpi-layer2 { 
    color: var(--accent-heritage); 
    transform: scale(1.04);
    text-shadow: 0 2px 20px rgba(45, 95, 63, 0.15);
}

.kpi-layer3 { 
    font-size: 0.85rem; 
    color: var(--ink-secondary); 
    line-height: 1.55; 
    font-weight: 500;
    letter-spacing: -0.005em;
}

/* ============================================
   SECTION TITLES — EDITORIAL PREMIUM
   ============================================ */
.section-title { 
    font-family: 'Fraunces', serif !important; 
    font-size: 1.75rem !important; 
    font-weight: 700 !important; 
    color: var(--accent-heritage) !important; 
    border-bottom: none;
    padding-bottom: 0.9rem; 
    margin-top: 3.5rem; 
    margin-bottom: 1.8rem; 
    display: inline-block; 
    letter-spacing: -0.025em;
    animation: slideInLeft 0.7s cubic-bezier(0.16, 1, 0.3, 1); 
    position: relative;
    font-variation-settings: "opsz" 144;
}

.section-title::after { 
    content: ''; 
    position: absolute; 
    bottom: 0;
    left: 0; 
    width: 100%; 
    height: 3px; 
    background: linear-gradient(90deg, var(--accent-heritage), var(--accent-copper), var(--accent-sand), transparent);
    background-size: 200% 100%;
    animation: borderFlow 4s linear infinite;
    border-radius: 2px;
}

.section-subtitle { 
    font-size: 0.95rem; 
    color: var(--ink-secondary); 
    margin-bottom: 2rem; 
    font-style: italic; 
    padding-left: 1.2rem; 
    border-left: 3px solid var(--accent-sand);
    font-weight: 400;
    letter-spacing: -0.005em;
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.2s backwards;
    font-family: 'Fraunces', serif;
}

/* ============================================
   INSIGHT / WATCHLIST / REC CARDS — PREMIUM
   ============================================ */
.insight-card { 
    background: linear-gradient(135deg, #EFF5EE 0%, #E5EFE4 100%);
    border-left: 4px solid var(--accent-heritage); 
    padding: 1.5rem 1.8rem; 
    border-radius: 4px var(--radius-lg) var(--radius-lg) 4px; 
    color: var(--ink-primary); 
    margin: 1.3rem 0; 
    line-height: 1.75; 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    font-size: 0.95rem; 
    box-shadow: var(--shadow-sm); 
    animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
    overflow: hidden;
}

.insight-card::before {
    content: '"';
    position: absolute;
    top: -20px; left: 15px;
    font-size: 6rem;
    font-family: 'Fraunces', serif;
    color: var(--accent-sage);
    opacity: 0.1;
    line-height: 1;
}

.insight-card:hover { 
    transform: translateX(10px); 
    box-shadow: var(--shadow-md); 
    border-left-width: 6px;
}

.watchlist-card { 
    background: linear-gradient(135deg, #FDF5EC 0%, #FAECD9 100%);
    border-left: 4px solid var(--accent-copper); 
    padding: 1.5rem 1.8rem; 
    border-radius: 4px var(--radius-lg) var(--radius-lg) 4px; 
    color: var(--ink-primary); 
    margin: 1.3rem 0; 
    line-height: 1.75; 
    font-size: 0.95rem; 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    box-shadow: var(--shadow-sm); 
    animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
}

.watchlist-card::before {
    content: '';
    position: absolute;
    top: 50%; right: -30px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, rgba(184, 115, 51, 0.1) 0%, transparent 70%);
    border-radius: 50%;
    transform: translateY(-50%);
}

.watchlist-card:hover { 
    transform: translateX(10px); 
    box-shadow: var(--shadow-md); 
    border-left-width: 6px;
}

.rec-card { 
    background: var(--bg-card); 
    border: 1px solid var(--border); 
    border-left: 4px solid var(--accent-sand); 
    padding: 1.4rem 1.7rem; 
    border-radius: 4px var(--radius-md) var(--radius-md) 4px; 
    color: var(--ink-primary); 
    margin: 0.9rem 0; 
    line-height: 1.7; 
    font-size: 0.93rem; 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    box-shadow: var(--shadow-xs); 
    animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
}

.rec-card:hover { 
    transform: translateX(10px); 
    box-shadow: var(--shadow-md); 
    border-left-width: 6px;
    border-color: var(--accent-sand);
}

.priority-card { 
    background: 
        linear-gradient(145deg, #FFFFFF 0%, #FAF7F0 100%);
    border: 1px solid var(--border); 
    padding: 2rem 1.7rem; 
    border-radius: var(--radius-lg); 
    color: var(--ink-primary); 
    margin: 0.7rem 0; 
    transition: all 0.5s cubic-bezier(0.16, 1, 0.3, 1); 
    text-align: center; 
    box-shadow: var(--shadow-sm); 
    animation: fadeInScale 0.7s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
    overflow: hidden;
}

.priority-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-heritage), var(--accent-copper), var(--accent-sand));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

.priority-card:hover::before { transform: scaleX(1); }

.priority-card:nth-child(1) { animation-delay: 0.1s; } 
.priority-card:nth-child(2) { animation-delay: 0.2s; } 
.priority-card:nth-child(3) { animation-delay: 0.3s; }

.priority-card:hover { 
    transform: translateY(-10px); 
    box-shadow: var(--shadow-lg); 
    border-color: var(--accent-heritage);
}

/* ============================================
   TABS — PREMIUM PILLS
   ============================================ */
.stTabs [data-baseweb="tab-list"] { 
    gap: 8px; 
    background: 
        linear-gradient(180deg, rgba(229, 223, 208, 0.5) 0%, rgba(229, 223, 208, 0.3) 100%);
    padding: 0.5rem; 
    border-radius: var(--radius-md); 
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    border: 1px solid rgba(229, 223, 208, 0.5);
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] { 
    background-color: transparent !important; 
    border-radius: var(--radius-sm) !important; 
    color: var(--ink-secondary) !important; 
    font-weight: 600 !important; 
    padding: 0.75rem 1.5rem !important; 
    border: none !important; 
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important; 
    font-size: 0.88rem !important;
    letter-spacing: 0.01em;
}

.stTabs [data-baseweb="tab"]:hover { 
    background-color: rgba(139, 168, 136, 0.12) !important; 
    color: var(--accent-heritage) !important; 
    transform: translateY(-2px) !important; 
}

.stTabs [aria-selected="true"] { 
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF7F0 100%) !important; 
    color: var(--accent-heritage) !important; 
    border: 1px solid var(--border) !important; 
    box-shadow: 
        var(--shadow-sm),
        0 2px 0 rgba(45, 95, 63, 0.08) !important; 
    font-weight: 700 !important;
}

/* ============================================
   BUTTONS — PREMIUM INTERACTIVE
   ============================================ */
.stButton > button, .stDownloadButton > button { 
    border-radius: var(--radius-md) !important; 
    font-weight: 600 !important; 
    padding: 0.7rem 1.7rem !important; 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important; 
    font-size: 0.9rem !important; 
    letter-spacing: 0.015em; 
    position: relative; 
    overflow: hidden;
    font-family: 'Inter', sans-serif;
}

.stButton > button { 
    background: linear-gradient(135deg, var(--accent-heritage) 0%, #3E7550 100%) !important; 
    color: #FFFFFF !important; 
    border: none !important; 
    box-shadow: 
        0 4px 12px rgba(45, 95, 63, 0.25),
        0 2px 4px rgba(45, 95, 63, 0.15) !important;
}

.stButton > button::before { 
    content: ''; 
    position: absolute; 
    top: 50%; 
    left: 50%; 
    width: 0; 
    height: 0; 
    border-radius: 50%; 
    background: rgba(255, 255, 255, 0.2); 
    transform: translate(-50%, -50%); 
    transition: width 0.6s ease, height 0.6s ease;
}

.stButton > button:hover::before { 
    width: 400px; 
    height: 400px; 
}

.stButton > button:hover { 
    background: linear-gradient(135deg, var(--accent-heritage-deep) 0%, var(--accent-heritage) 100%) !important; 
    transform: translateY(-4px) !important; 
    box-shadow: 
        0 12px 28px rgba(45, 95, 63, 0.35),
        0 4px 8px rgba(45, 95, 63, 0.2) !important;
}

.stDownloadButton > button { 
    background-color: var(--bg-card) !important; 
    color: var(--accent-heritage) !important; 
    border: 1.5px solid var(--accent-heritage) !important;
    font-weight: 600 !important;
}

.stDownloadButton > button:hover { 
    background: linear-gradient(135deg, #F0F5EE 0%, #E8F0E5 100%) !important; 
    transform: translateY(-4px) !important; 
    box-shadow: 0 12px 28px rgba(45, 95, 63, 0.15) !important;
    border-color: var(--accent-heritage-deep) !important;
}

/ Form labels /
.stSelectbox label, .stSlider label, .stRadio label, .stMultiSelect label, .stNumberInput label { 
    color: var(--ink-secondary) !important; 
    font-weight: 600 !important; 
    font-size: 0.85rem !important; 
    letter-spacing: 0.01em; 
    transition: color 0.3s ease;
}

.stSelectbox label:hover, .stSlider label:hover, .stRadio label:hover { 
    color: var(--accent-heritage) !important; 
}

/* ============================================
   METRICS — PREMIUM CARDS
   ============================================ */
div[data-testid="stMetric"] { 
    background: 
        linear-gradient(145deg, #FFFFFF 0%, #FAF7F0 100%);
    padding: 1.2rem 1.4rem; 
    border-radius: var(--radius-md); 
    border: 1px solid var(--border); 
    box-shadow: var(--shadow-sm); 
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1); 
    animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
    position: relative;
    overflow: hidden;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent-heritage), var(--accent-copper));
    opacity: 0;
    transition: opacity 0.3s ease;
}

div[data-testid="stMetric"]:hover::before { opacity: 1; }

div[data-testid="stMetric"]:hover { 
    transform: translateY(-6px); 
    box-shadow: var(--shadow-md); 
    border-color: var(--accent-sage);
}

div[data-testid="stMetric"] label { 
    color: var(--ink-muted) !important; 
    font-family: 'Inter', sans-serif !important; 
    font-weight: 600;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
    color: var(--ink-primary) !important; 
    font-family: 'Fraunces', serif !important; 
    font-weight: 700 !important; 
    font-size: 1.65rem;
    letter-spacing: -0.025em;
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover div[data-testid="stMetricValue"] { 
    color: var(--accent-heritage) !important; 
    transform: scale(1.04);
}

/ Header /
header[data-testid="stHeader"] { 
    background: rgba(250, 247, 240, 0.85) !important; 
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 0 rgba(255,255,255,0.5) inset;
}

/ Dataframe /
.stDataFrame { 
    border-radius: var(--radius-md) !important; 
    border: 1px solid var(--border) !important; 
    overflow: hidden; 
    box-shadow: var(--shadow-sm); 
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1); 
    transition: all 0.4s ease;
}

.stDataFrame:hover { 
    box-shadow: var(--shadow-md); 
    border-color: var(--accent-sage);
}

/ Warning /
.stWarning { 
    border-radius: var(--radius-md) !important; 
    border-left: 4px solid var(--accent-copper) !important; 
    background: #FDF5EC !important; 
    color: var(--ink-primary) !important; 
    padding: 1.1rem 1.3rem !important; 
    font-size: 0.9rem; 
    animation: slideInLeft 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: var(--shadow-sm);
}

/ Divider /
.organic-divider { 
    height: 1px; 
    background: linear-gradient(90deg, transparent, var(--border-strong), var(--accent-sand), var(--border-strong), transparent); 
    margin: 3.5rem 0; 
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}

.organic-divider::before {
    content: '🌿';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.2rem;
    background: var(--bg-paper);
    padding: 0 1rem;
    opacity: 0.5;
}

/ Code blocks /
.stCodeBlock { 
    border-radius: var(--radius-md) !important; 
    border: 1px solid var(--border-strong) !important; 
    background: var(--ink-primary) !important; 
    animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    font-family: 'JetBrains Mono', monospace;
}

/ Aggregate badge /
.aggregate-badge { 
    display: inline-block; 
    background: linear-gradient(135deg, var(--accent-copper) 0%, var(--accent-copper-light) 100%);
    color: #FFFFFF; 
    padding: 0.4rem 1.1rem; 
    border-radius: 999px; 
    font-size: 0.72rem; 
    font-weight: 700; 
    text-transform: uppercase; 
    letter-spacing: 0.1em; 
    margin-left: 0.9rem; 
    box-shadow: 
        0 4px 12px rgba(184, 115, 51, 0.3),
        0 0 0 1px rgba(255,255,255,0.1) inset;
    animation: pulse 2.5s ease-in-out infinite;
    vertical-align: middle;
}

/ Plotly chart container enhancement /
.js-plotly-plot .plotly {
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    background: var(--bg-card);
    border: 1px solid var(--border);
}

.js-plotly-plot .plotly:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-3px);
    border-color: var(--accent-sage);
}

/ Floating botanical elements /
.floating-botanicals {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}

.botanical {
    position: absolute;
    opacity: 0.08;
    font-size: 2rem;
    animation: leafFall linear infinite;
}

.botanical:nth-child(1) { left: 5%; animation-duration: 25s; animation-delay: 0s; }
.botanical:nth-child(2) { left: 25%; animation-duration: 30s; animation-delay: 5s; }
.botanical:nth-child(3) { left: 50%; animation-duration: 28s; animation-delay: 10s; }
.botanical:nth-child(4) { left: 75%; animation-duration: 32s; animation-delay: 3s; }
.botanical:nth-child(5) { left: 95%; animation-duration: 26s; animation-delay: 7s; }

/ Selection styling /
::selection {
    background: var(--accent-sage);
    color: white;
}

/ Focus rings /
*:focus-visible {
    outline: 2px solid var(--accent-heritage);
    outline-offset: 2px;
}

/* ============================================
   RESPONSIVE ENHANCEMENTS
   ============================================ */
@media (max-width: 1024px) {
    .hero-title { font-size: 2.4rem !important; }
    .hero-strip { padding: 2.5rem 2rem; }
    .hero-mini-stats { gap: 2rem; }
    .mini-stat-value { font-size: 1.5rem; }
}

@media (max-width: 768px) {
    .hero-title { font-size: 1.9rem !important; }
    .hero-subtitle { max-width: 100%; font-size: 0.95rem; }
    .hero-mini-stats { gap: 1.5rem; }
    .kpi-layer2 { font-size: 1.8rem; }
}

    🍃
    🌿
    🍂
    🌱
    ☘️

""", unsafeallowhtml=True)

=========================================================
DATASET (BPS 2025)
=========================================================
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
    return pd.readcsv(io.StringIO(CSVDATA))

df = load_data()
numeric_cols = [c for c in df.columns if c != "Provinsi"]

=========================================================
COMMODITY IDENTITY MAP
=========================================================
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

def getcommattr(comm, attr="color"):
    return COMMODITY_IDENTITY.get(comm, {}).get(attr, "#6B9278")

def format_num(x):
    try:
        v = float(x)
        if v >= 1000000: return f"{v/1000000:.1f}M"
        if v >= 1000: return f"{v/1000:.1f}K"
        return f"{v:,.0f}"
    except: return str(x)

def format_ton(x):
    try: return f"{float(x):,.2f}"
    except: return str(x)

def createintelkpi(label, value, subtext, icon, color):
    return f'''
        {icon} {label}
        {value}
        {subtext}
    '''

def applyplantationlayout(fig, height=480):
    fig.update_layout(
        template="plotlywhite", paperbgcolor="#FAF7F0", plot_bgcolor="#FFFFFF",
        font=dict(color="#1A2B20", family="Inter, sans-serif", size=13),
        height=height, margin=dict(l=40, r=30, t=60, b=50),
        xaxis=dict(gridcolor="#E5DFD0", zerolinecolor="#E5DFD0", tickfont=dict(size=11, color="#3E5245", family="Inter")),
        yaxis=dict(gridcolor="#E5DFD0", zerolinecolor="#E5DFD0", tickfont=dict(size=11, color="#3E5245", family="Inter")),
        title=dict(font=dict(size=17, color="#2D5F3F", family="Fraunces, serif", weight=600), x=0.02, xanchor="left"),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#2D5F3F", font=dict(color="#1A2B20", family="Inter", size=13)),
        legend=dict(font=dict(color="#3E5245", size=12))
    )
    return fig

=========================================================
SIDEBAR
=========================================================
with st.sidebar:
    st.markdown('''
    
        🌿
        Plantation Intel
        Indonesia 2025
    
    ''', unsafeallowhtml=True)
    
    st.markdown('🧭 Navigasi', unsafeallowhtml=True)
    menu = st.radio("Menu",
        ["🏠 Ringkasan Nasional", "🌴 Profil Komoditas", "🗺️ Profil Provinsi",
         "🔬 Eksplorasi Visual", "📊 Analisis Produksi", "🌍 Sebaran Wilayah",
         "📈 Proyeksi & Model", "🧠 Insight & Strategi", "📦 Data & Ekspor"],
        label_visibility="collapsed")
    st.markdown('', unsafeallowhtml=True)
    
    st.markdown('🎛️ Filter Data', unsafeallowhtml=True)
    
    commodityoptions = ["🌾 Semua Komoditas"] + numericcols
    selectedcommodityraw = st.selectbox("Komoditas Fokus", commodityoptions, index=0, key="sidecomm")
    
    isallcommodities = (selectedcommodityraw == "🌾 Semua Komoditas")
    selectedcommodity = "TotalOutput" if isallcommodities else selectedcommodityraw
    commoditydisplayname = "Semua Komoditas (Total Output)" if isallcommodities else selectedcommodityraw
    
    selectedprovince = st.selectbox("Wilayah Provinsi", ["Semua Provinsi"] + df["Provinsi"].tolist(), index=0, key="sideprov")
    topn = st.slider("Top N Sentra", 5, 20, 10, key="sidetopn")
    showzeros = st.checkbox("Tampilkan wilayah tanpa produksi", value=True, key="sidezero")
    st.markdown('', unsafeallowhtml=True)
    
    if not isallcommodities:
        comminfo = COMMODITYIDENTITY[selected_commodity]
        st.markdown(f'''
            🌱 Commodity Brief
            
                {comminfo["icon"]} {selectedcommodity}
                {comm_info["sector"]}
                {comm_info["desc"]}
            
        ''', unsafeallowhtml=True)
    else:
        st.markdown('''
            🌾 Mode Agregat Aktif
            
                📊 Total Output
                Semua Komoditas
                Menampilkan gabungan total produksi 7 komoditas perkebunan utama Indonesia.
            
        ''', unsafeallowhtml=True)
    
    st.markdown('''
    
        🌴
        Tropical Heritage
    
    ''', unsafeallowhtml=True)

active_df = df.copy()
if selected_province != "Semua Provinsi":
    activedf = activedf[activedf["Provinsi"] == selectedprovince].copy()
if not show_zeros:
    activedf = activedf[(activedf[numericcols].sum(axis=1) > 0)].copy()

activedf["TotalOutput"] = activedf[numericcols].sum(axis=1)

=========================================================
PAGE 1: RINGKASAN NASIONAL
=========================================================
if menu == "🏠 Ringkasan Nasional":
    totalprod = activedf[numeric_cols].sum().sum()
    activeprovs = len(activedf)

    st.markdown(f"""
    
        Plantation Intelligence
        Pusat intelijen strategis untuk pemetaan sentra produksi, portofolio komoditas andalan, dan analisis struktur perkebunan nasional per provinsi.
        
            🌾 Sentra Produksi
            📊 Portofolio Komoditas
            🗺️ Intelijen Wilayah
            📈 Proyeksi Panen
        
        
            Provinsi{active_provs}
            Komoditas7
            Tahun Data2025
            Total Output{formatnum(totalprod)} Ton
        
    
    """, unsafeallowhtml=True)

    commtotals = activedf[numericcols].sum().sortvalues(ascending=False)
    domcomm = commtotals.index[0] if not comm_totals.empty else "-"
    domval = commtotals.iloc[0] if not comm_totals.empty else 0
    domshare = (domval / totalprod * 100) if totalprod > 0 else 0

    if not active_df.empty:
        topprov = activedf.loc[activedf[numericcols].sum(axis=1).idxmax(), "Provinsi"]
        topval = activedf[numeric_cols].sum(axis=1).max()
        diverseidx = (activedf[numeric_cols] > 0).sum(axis=1).idxmax()
        diverseprov = activedf.loc[diverse_idx, "Provinsi"]
        diversecount = int((activedf.loc[diverseidx, numericcols] > 0).sum())
        top5share = (activedf[numericcols].sum(axis=1).nlargest(5).sum() / max(1, totalprod) * 100)
    else:
        topprov = diverseprov = "-"
        topval = diversecount = top5_share = 0

    c1, c2, c3, c4 = st.columns(4)
    if isallcommodities:
        with c1: st.markdown(createintelkpi("Mode Analisis", "Agregat", "Seluruh komoditas", "🌾", "#B87333"), unsafeallowhtml=True)
    else:
        with c1: st.markdown(createintelkpi("Komoditas Dominan", domcomm, f"Menyumbang {domshare:.1f}%", getcommattr(domcomm,"icon"), getcommattr(domcomm,"colorlight")), unsafeallow_html=True)
    with c2: st.markdown(createintelkpi("Sentra Tertinggi", topprov[:14], f"Output {formatton(topval)}", "🏆", "#B87333"), unsafeallow_html=True)
    with c3: st.markdown(createintelkpi("Terdiversifikasi", diverseprov[:14], f"{diversecount} komoditas", "🌱", "#2D5F3F"), unsafeallowhtml=True)
    with c4: st.markdown(createintelkpi("Konsentrasi Top-5", f"{top5share:.1f}%", "Pangsa 5 provinsi", "🎯", "#8BA888"), unsafeallow_html=True)

    st.markdown(f'🌾 Sentra & Kontribusi Produksi Nasional{"Mode Agregat" if isallcommodities else ""}', unsafeallowhtml=True)
    
    nattab1, nattab2 = st.tabs(["📊 Grafik Sentra", "🗺️ Peta Geospasial"])
    
    with nat_tab1:
        coll, colr = st.columns(2)
        with col_l:
            topdf = activedf[["Provinsi", selectedcommodity]].sortvalues(selectedcommodity, ascending=False).head(topn)
            barcolor = "#B87333" if isallcommodities else getcommattr(selectedcommodity, "color_light")
            fig1 = px.bar(topdf[::-1], x=selectedcommodity, y="Provinsi", orientation='h', text=selected_commodity,
                          title=f"Sentra Produksi: {commoditydisplayname}", colordiscretesequence=[bar_color])
            fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotlychart(applyplantationlayout(fig1, 520), usecontainerwidth=True, key="natbar_01")
        with col_r:
            provtotals = activedf.copy()
            provtotals["TotalOutput"] = provtotals[numericcols].sum(axis=1)
            topprovdf = provtotals.nlargest(topn, "TotalOutput")[["Provinsi", "TotalOutput"]].sortvalues("TotalOutput", ascending=True)
            fig2 = px.bar(topprovdf, x="TotalOutput", y="Provinsi", orientation='h', text="TotalOutput",
                          title="Kontribusi Output Perkebunan per Provinsi", colordiscretesequence=["#B87333"])
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotlychart(applyplantationlayout(fig2, 520), usecontainerwidth=True, key="natbar_02")
    
    with nat_tab2:
        st.markdown("""
        
            🗺️ Peta Geospasial: Visualisasi spasial menunjukkan koridor produksi perkebunan Indonesia. 
            Hover pada peta untuk melihat detail produksi tiap provinsi. 
            Warna lebih gelap = produksi lebih tinggi.
        
        """, unsafeallowhtml=True)
        
        map_metric = st.radio(
            "Pilih Metrik Peta:",
            ["Produksi Komoditas Terpilih", "Total Output Semua Komoditas"],
            horizontal=True, key="mapmetricselector"
        )
        
        if map_metric == "Produksi Komoditas Terpilih":
            mapdf = activedf[["Provinsi", selected_commodity]].copy()
            mapdf = mapdf[mapdf[selectedcommodity] > 0]
            maptitle = f"Sebaran Produksi {commoditydisplay_name} di Indonesia"
            mapvaluecol = selected_commodity
            maplabel = f"{commoditydisplay_name} (Ribu Ton)"
            isaggmap = isallcommodities
        else:
            mapdf = activedf.copy()
            mapdf["TotalOutputMap"] = mapdf[numeric_cols].sum(axis=1)
            mapdf = mapdf[mapdf["TotalOutput_Map"] > 0]
            map_title = "Total Output Perkebunan di Indonesia"
            mapvaluecol = "TotalOutputMap"
            map_label = "Total Output (Ribu Ton)"
            isaggmap = True
        
        if not map_df.empty:
            figmap = createchoroplethmap(mapdf, mapvaluecol, maptitle, maplabel, isaggregate=isagg_map)
            if fig_map:
                st.plotlychart(figmap, usecontainerwidth=True, key="natchoroplethmain")
                
                colm1, colm2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(createintelkpi("Provinsi Terpetakan", str(len(mapdf)), "Dengan produksi > 0", "📍", "#6B9278"), unsafeallow_html=True)
                with col_m2:
                    maxprov = mapdf.loc[mapdf[mapvalue_col].idxmax(), "Provinsi"]
                    maxval = mapdf[mapvaluecol].max()
                    st.markdown(createintelkpi("Hotspot Tertinggi", maxprov[:14], f"{formatton(maxval)} ribu ton", "🔥", "#B87333"), unsafeallow_html=True)
                with col_m3:
                    medianval = mapdf[mapvaluecol].median()
                    st.markdown(createintelkpi("Median Produksi", formatton(medianval), "Tendensi sentral", "⚖️", "#8BA888"), unsafeallowhtml=True)
        else:
            st.warning("⚠️ Tidak ada data produksi untuk ditampilkan pada peta.")
    
    st.markdown('', unsafeallowhtml=True)
    st.markdown('🗺️ Peta Struktur Komoditas Nasional', unsafeallowhtml=True)
    colt1, colt2 = st.columns(2)
    with col_t1:
        compdf = activedf[numericcols].sum().resetindex()
        comp_df.columns = ["Komoditas", "Produksi"]
        figpie = px.pie(compdf, values="Produksi", names="Komoditas", hole=0.65,
                         color="Komoditas", colordiscretemap={c: getcommattr(c, "colorlight") for c in compdf["Komoditas"]},
                         title="Komposisi Output Nasional")
        figpie.updatetraces(textinfo='percent+label', textfontsize=12, textfontcolor='#1A2B20')
        st.plotlychart(applyplantationlayout(figpie, 450), usecontainerwidth=True, key="natpie01")
    with col_t2:
        tmdf = activedf.copy()
        tmdf["Total"] = tmdf[numeric_cols].sum(axis=1)
        tmdf = tmdf[tm_df["Total"] > 0].nlargest(15, "Total")
        figtm = px.treemap(tmdf, path=["Provinsi"], values="Total", color="Total", colorcontinuousscale=[[0, "#E5DFD0"], [0.5, "#8BA888"], [1, "#2D5F3F"]],
                            title="15 Wilayah Teratas (Treemap)")
        figtm.updatetraces(textfontcolor="#FFFFFF", textfontsize=13)
        st.plotlychart(applyplantationlayout(figtm, 450), usecontainerwidth=True, key="nattree01")

    st.markdown('⚠️ Watchlist Strategis', unsafeallowhtml=True)
    st.markdown(f"""
    
        📊 Poin Kritis:
        • Konsentrasi Spasial: Top 5 provinsi menguasai {top5_share:.1f}% — mitigasi risiko iklim & pasar.
        • Dominasi {dom_comm}: Penopang devisa, namun rentan fluktuasi harga global.
        • Model Diversifikasi: {diverse_prov} menunjukkan portofolio seimbang.
        • Kesenjangan: Teh & Tebu sebaran sangat sempit, butuh klasterisasi spesifik lokasi.
    
    """, unsafeallowhtml=True)

=========================================================
PAGE 2: PROFIL KOMODITAS
=========================================================
elif menu == "🌴 Profil Komoditas":
    if isallcommodities:
        st.markdown("""
        
            ℹ️ Mode Agregat Aktif: Halaman ini menampilkan profil per komoditas tunggal. 
            Silakan pilih komoditas spesifik di sidebar untuk analisis mendalam, atau gunakan filter di bawah ini.
        
        """, unsafeallowhtml=True)
    
    target = st.selectbox("Pilih Komoditas untuk Analisis Mendalam", numericcols, key="profcomm_sel")
    info = COMMODITY_IDENTITY[target]

    st.markdown(f"""
    
        {info['icon']}
        
        {info['icon']} Profil Komoditas: {target}
        Sektor: {info['sector']}Potret: {info['desc']}
    
    """, unsafeallowhtml=True)

    cdf = activedf[["Provinsi", target]].copy()
    cdf = cdf[cdf[target] > 0].sortvalues(target, ascending=False)
    tot = c_df[target].sum()
    topp = cdf.iloc[0]["Provinsi"] if not c_df.empty else "-"
    topv = cdf.iloc[0][target] if not c_df.empty else 0
    med = cdf[target].median() if not cdf.empty else 0
    share5 = cdf.head(5)[target].sum()/max(1,tot)*100 if not cdf.empty else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(createintelkpi("Total Panen", formatnum(tot), "Volume agregat", info["icon"], info["colorlight"]), unsafeallowhtml=True)
    with c2: st.markdown(createintelkpi("Sentra Utama", topp[:14], "Kontributor terbesar", "🏆", "#B87333"), unsafeallow_html=True)
    with c3: st.markdown(createintelkpi("Median Produksi", formatton(med), "Tendensi sentral", "⚖️", "#8BA888"), unsafeallow_html=True)
    with c4: st.markdown(createintelkpi("Konsentrasi Top-5", f"{share5:.1f}%", "Pangsa 5 provinsi", "🎯", "#6B9278"), unsafeallowhtml=True)

    tab1, tab2, tab3 = st.tabs(["🏆 Hirarki Sentra", "📊 Potret Distribusi", "🗺️ Peta Kontribusi"])
    commcolor = getcommattr(target, "colorlight")
    with tab1:
        fig = px.bar(cdf.head(topn)[::-1], x=target, y="Provinsi", orientation='h', text=target,
                     title=f"Hirarki Sentra Produksi: {target}", colordiscretesequence=[comm_color])
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotlychart(applyplantationlayout(fig, 600), usecontainerwidth=True, key=f"commbar_{target}")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            figh = px.histogram(cdf, x=target, nbins=15, title="Distribusi Volume", colordiscretesequence=[comm_color])
            st.plotlychart(applyplantationlayout(figh, 480), usecontainerwidth=True, key=f"commhist{target}")
        with c2:
            figb = px.box(cdf, y=target, points="all", title="Persebaran & Sentra Ekstrem", colordiscretesequence=[comm_color])
            st.plotlychart(applyplantationlayout(figb, 480), usecontainerwidth=True, key=f"commbox{target}")
    with tab3:
        figt = px.treemap(cdf.head(15), path=["Provinsi"], values=target, color=target,
                           colorcontinuousscale=[[0, "#E5DFD0"], [0.5, info["color_light"]], [1, info["color"]]], title="Treemap Kontribusi Wilayah")
        figt.updatetraces(textfontcolor="#FFFFFF", textfontsize=13)
        st.plotlychart(applyplantationlayout(figt, 500), usecontainerwidth=True, key=f"commtree{target}")
        figp = px.pie(cdf.head(10), values=target, names="Provinsi", hole=0.65, title="Komposisi 10 Wilayah Teratas",
                       colordiscretesequence=SOFT_PALETTE)
        figp.updatetraces(textinfo='percent+label')
        st.plotlychart(applyplantationlayout(figp, 450), usecontainerwidth=True, key=f"commpie{target}")

    conc_note = "terkonsentrasi kuat" if share5 > 60 else "relatif terdistribusi"
    st.markdown(f"""
    
        {info['icon']} Intelijen {target}: Produksi bersifat {conc_note} ({share5:.1f}% dikuasai Top-5).
        {top_p} berperan sebagai anchor. Potensi hilirisasi perlu difokuskan di wilayah penyangga.
    
    """, unsafeallowhtml=True)

=========================================================
PAGE 3: PROFIL PROVINSI
=========================================================
elif menu == "🗺️ Profil Provinsi":
    targetprov = st.selectbox("Pilih Provinsi", df["Provinsi"].tolist(), key="profprov_sel")
    prow = df[df["Provinsi"] == targetprov].iloc[0]
    pdata = {c: prow[c] for c in numeric_cols}
    totp = sum(pdata.values())
    activec = sum(1 for v in pdata.values() if v > 0)
    domc = max(pdata, key=pdata.get) if totp > 0 else "-"
    domv = pdata[domc] if totp > 0 else 0
    domshare = (domv/totp*100) if totp > 0 else 0
    natshare = (totp / df[numericcols].sum().sum() * 100) if df[numericcols].sum().sum() > 0 else 0

    st.markdown(f"""
    
        🗺️
        
        🗺️ Profil Perkebunan: {target_prov}
        Total output {formatton(totp)} ribu ton ({natshare:.2f}% nasional). Komoditas andalan: {domc} ({dom_share:.1f}%).
    
    """, unsafeallowhtml=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(createintelkpi("Total Output", formatnum(totp), "Agregat komoditas", "📦", "#6B9278"), unsafeallowhtml=True)
    with c2: st.markdown(createintelkpi("Komoditas Andalan", domc, f"Penopang ({domshare:.0f}%)", getcommattr(domc,"icon"), getcommattr(domc,"colorlight")), unsafeallow_html=True)
    with c3: st.markdown(createintelkpi("Sektor Aktif", str(activec), f"dari {len(numericcols)}", "🌱", "#8BA888"), unsafeallowhtml=True)
    with c4: st.markdown(createintelkpi("Tingkat Dominasi", f"{domshare:.0f}%", "Konsentrasi utama", "🎯", "#B87333"), unsafeallow_html=True)

    st.markdown('🌳 Struktur & Portofolio Komoditas', unsafeallowhtml=True)
    col1, col2 = st.columns(2)
    with col1:
        pdf = pd.DataFrame({"Komoditas": list(pdata.keys()), "Produksi": list(pdata.values())}).sortvalues("Produksi", ascending=False)
        fig = px.bar(p_df, x="Komoditas", y="Produksi", text="Produksi", color="Komoditas",
                     colordiscretemap={c: getcommattr(c, "colorlight") for c in pdf["Komoditas"]}, title="Portofolio Komoditas")
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotlychart(applyplantationlayout(fig, 460), usecontainerwidth=True, key=f"provbar{targetprov}")
    with col2:
        fig2 = px.pie(pdf[pdf["Produksi"]>0], values="Produksi", names="Komoditas", hole=0.65, title="Komposisi Output Daerah",
                      colordiscretesequence=SOFT_PALETTE)
        fig2.update_traces(textinfo='percent+label')
        st.plotlychart(applyplantationlayout(fig2, 460), usecontainerwidth=True, key=f"provpie{targetprov}")

    st.markdown('', unsafeallowhtml=True)
    st.markdown('🎯 Radar Portofolio vs Nasional', unsafeallowhtml=True)
    colr1, colr2 = st.columns([1, 1])
    with col_r1:
        natavg = {c: df[c].mean() for c in numericcols}
        pvalues = [pdata[c] for c in numeric_cols]
        pmax = max(pvalues) if p_values else 1
        pnorm = [(v / pmax * 100) if pmax > 0 else 0 for v in pvalues]
        natvalues = list(natavg.values())
        natmax = max(natvalues) if nat_values else 1
        natnorm = [(v / natmax * 100) if natmax > 0 else 0 for v in natvalues]
        fig_r = go.Figure()
        figr.addtrace(go.Scatterpolar(r=pnorm + [pnorm[0]], theta=numericcols + [numericcols[0]],
                                        fill='toself', name=targetprov, linecolor="#2D5F3F", fillcolor="rgba(45, 95, 63, 0.3)"))
        figr.addtrace(go.Scatterpolar(r=natnorm + [natnorm[0]], theta=numericcols + [numericcols[0]],
                                        fill='toself', name="Rata-rata Nasional", line_color="#B87333", fillcolor="rgba(184, 115, 51, 0.15)"))
        figr.updatelayout(polar=dict(bgcolor="#FFFFFF",
                                       radialaxis=dict(visible=True, showticklabels=False, gridcolor="#E5DFD0"),
                                       angularaxis=dict(gridcolor="#E5DFD0", tickfont=dict(color="#3E5245", family="Inter"))),
                            showlegend=True, title="Radar (Normalisasi)")
        st.plotlychart(applyplantationlayout(figr, 500), usecontainerwidth=True, key=f"provradar{target_prov}")
    with col_r2:
        benchdf = pd.DataFrame({"Komoditas": numericcols, "Provinsi": pvalues, "Nasional": list(natavg.values())})
        benchdf["Deviasi"] = benchdf["Provinsi"] - bench_df["Nasional"]
        colors = ["#6B9278" if v >= 0 else "#B39471" for v in bench_df["Deviasi"]]
        fig3 = go.Figure(go.Bar(y=benchdf["Komoditas"], x=benchdf["Deviasi"], orientation='h', marker_color=colors))
        fig3.updatelayout(title="Deviasi vs Rata-rata Nasional", xaxistitle="Deviasi (Ribu Ton)")
        st.plotlychart(applyplantationlayout(fig3, 500), usecontainerwidth=True, key=f"provbench{targetprov}")

    depnote = "⚠️ Sangat bergantung pada satu komoditas." if domshare > 70 else "✅ Portofolio relatif seimbang."
    st.markdown(f'🗺️ {targetprov}: {activec} komoditas aktif. {depnote} Strategi: perkuat klaster unggulan & hilirisasi lokal.', unsafeallow_html=True)

=========================================================
PAGE 4: EKSPLORASI VISUAL
=========================================================
elif menu == "🔬 Eksplorasi Visual":
    st.markdown(f'🔬 Eksplorasi Visual & EDA{"Mode Agregat" if isallcommodities else ""}', unsafeallowhtml=True)
    st.markdown('5 perspektif visual: Overview, Distribusi, Hubungan, Korelasi, Deep Dive', unsafeallowhtml=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Overview", "📊 Distribusi", "🔗 Hubungan", "🌡️ Korelasi", "📍 Deep Dive"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            topdf = activedf[["Provinsi", selectedcommodity]].sortvalues(selectedcommodity, ascending=False).head(topn)
            barcolor = "#B87333" if isallcommodities else getcommattr(selectedcommodity,"color_light")
            fig = px.bar(topdf[::-1], x=selectedcommodity, y="Provinsi", orientation='h', text=selected_commodity,
                         title=f"Top Sentra {commoditydisplayname}", colordiscretesequence=[bar_color])
            fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotlychart(applyplantationlayout(fig, 500), usecontainerwidth=True, key="edaoverview_1")
        with c2:
            temp = activedf.copy(); temp["Total"] = temp[numericcols].sum(axis=1)
            toptotal = temp.nlargest(topn, "Total").sort_values("Total", ascending=True)
            fig2 = px.bar(toptotal, x="Total", y="Provinsi", orientation='h', text="Total", title="Top Provinsi Total Produksi", colordiscrete_sequence=["#B87333"])
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotlychart(applyplantationlayout(fig2, 500), usecontainerwidth=True, key="edaoverview_2")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            barcolor = "#B87333" if isallcommodities else getcommattr(selectedcommodity,"color_light")
            fig = px.histogram(activedf, x=selectedcommodity, nbins=15, title=f"Distribusi {commoditydisplayname}",
                               colordiscretesequence=[bar_color])
            st.plotlychart(applyplantationlayout(fig, 480), usecontainerwidth=True, key="edadist_1")
        with c2:
            melted = activedf.melt(idvars="Provinsi", valuevars=numericcols, varname="Komoditas", valuename="Produksi")
            fig2 = px.box(melted, x="Komoditas", y="Produksi", color="Komoditas", title="Persebaran Antar Komoditas",
                          colordiscretemap={c: getcommattr(c, "colorlight") for c in numericcols})
            fig2.updatexaxes(tickangle=35); fig2.updatelayout(showlegend=False)
            st.plotlychart(applyplantationlayout(fig2, 480), usecontainerwidth=True, key="edadist_2")
    with tab3:
        c1, c2 = st.columns(2)
        xv = c1.selectbox("Komoditas X", numericcols, index=0, key="edarelx")
        yv = c2.selectbox("Komoditas Y", numericcols, index=1, key="edarely")
        if xv != yv:
            fig = px.scatter(activedf, x=xv, y=yv, hovername="Provinsi", title=f"{xv} vs {yv}", colordiscretesequence=["#2D5F3F"])
            st.plotlychart(applyplantationlayout(fig, 550), usecontainerwidth=True, key="edarel_scatter")
            corr = activedf[[xv, y_v]].corr().iloc[0,1]
            strength = "kuat" if abs(corr)>0.6 else "sedang" if abs(corr)>0.3 else "lemah"
            st.markdown(f'🔗 Korelasi: {corr:.3f} ({strength})', unsafeallowhtml=True)
        else: st.warning("Pilih dua komoditas berbeda.")
    with tab4:
        corrm = activedf[numeric_cols].corr()
        fig = px.imshow(corrm, textauto=".2f", aspect="auto", zmin=-1, zmax=1,
                        colorcontinuousscale=[[0,"#A67B66"],[0.5,"#FAF7F0"],[1,"#2D5F3F"]],
                        title="Matriks Korelasi Agro-Ekologis")
        fig.update_traces(textfont=dict(color="#1A2B20", size=12, family="Inter"), xgap=3, ygap=3)
        fig.updatexaxes(side="bottom", tickangle=30); fig.updateyaxes(autorange="reversed")
        fig.updatelayout(paperbgcolor="#FAF7F0", plot_bgcolor="#FFFFFF",
                          font=dict(color="#1A2B20"), margin=dict(l=40,r=30,t=60,b=40))
        fig.update_coloraxes(colorbar=dict(title=dict(text="Korelasi", font=dict(color="#3E5245")),
                                            tickfont=dict(color="#3E5245"), len=0.75))
        st.plotlychart(fig, usecontainerwidth=True, key="edacorr")
    with tab5:
        deepprov = st.selectbox("Pilih Provinsi", activedf["Provinsi"].tolist(), key="eda_deep")
        pdf = activedf[activedf["Provinsi"]==deepprov]
        if not p_df.empty:
            row = p_df.iloc[0]
            profile = pd.DataFrame({"Komoditas": numericcols, "Produksi": [row[c] for c in numericcols]}).sort_values("Produksi", ascending=False)
            c1, c2 = st.columns([1.1, 1])
            with c1:
                fig = px.bar(profile, x="Komoditas", y="Produksi", text="Produksi", color="Komoditas",
                             colordiscretemap={c: getcommattr(c, "colorlight") for c in numericcols}, title=f"Profil {deep_prov}")
                fig.updatetraces(texttemplate='%{text:,.0f}', textposition='outside'); fig.updatelayout(showlegend=False)
                st.plotlychart(applyplantationlayout(fig, 500), usecontainerwidth=True, key=f"edadeepbar{deep_prov}")
            with c2:
                total_p = profile["Produksi"].sum()
                dom_comm = profile.iloc[0]["Komoditas"]
                st.markdown(createintelkpi("Total Produksi", formatnum(totalp), "Ribu Ton", "📦", "#6B9278"), unsafeallowhtml=True)
                st.markdown(createintelkpi("Komoditas Dominan", domcomm, formatton(profile.iloc[0]['Produksi']), getcommattr(domcomm,'icon'), getcommattr(domcomm,'colorlight')), unsafeallow_html=True)

=========================================================
PAGE 5: ANALISIS PRODUKSI
=========================================================
elif menu == "📊 Analisis Produksi":
    st.markdown('📊 Analisis Produksi & Perbandingan', unsafeallowhtml=True)
    tab1, tab2, tab3 = st.tabs(["⚔️ Duel Wilayah", "🔗 Relasi Komoditas", "🎯 Benchmarking"])
    with tab1:
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("Wilayah A", df["Provinsi"].tolist(), index=0, key="duel_a")
        p2 = c2.selectbox("Wilayah B", df["Provinsi"].tolist(), index=3, key="duel_b")
        d1, d2 = df[df["Provinsi"]==p1].iloc[0], df[df["Provinsi"]==p2].iloc[0]
        comp = pd.DataFrame({"Komoditas": numericcols, p1: [d1[c] for c in numericcols], p2: [d2[c] for c in numeric_cols]})
        fig = go.Figure()
        fig.addtrace(go.Bar(name=p1, x=comp["Komoditas"], y=comp[p1], markercolor="#6B9278"))
        fig.addtrace(go.Bar(name=p2, x=comp["Komoditas"], y=comp[p2], markercolor="#B87333"))
        fig.update_layout(barmode='group', title=f"{p1} vs {p2}")
        st.plotlychart(applyplantationlayout(fig, 480), usecontainerwidth=True, key="duelchart")
        w1, w2 = (comp[p1] >comp[p2]).sum(), (comp[p2] >comp[p1]).sum()
        st.markdown(f'⚔️ {p1} unggul {w1} komoditas | {p2} unggul {w2} komoditas.', unsafeallowhtml=True)
    with tab2:
        c1, c2 = st.columns(2)
        xv = c1.selectbox("X", numericcols, index=0, key="rel_x")
        yv = c2.selectbox("Y", numericcols, index=1, key="rel_y")
        if xv != yv:
            fig = px.scatter(activedf, x=xv, y=yv, hovername="Provinsi", title=f"{xv} vs {yv}", colordiscretesequence=["#2D5F3F"])
            st.plotlychart(applyplantationlayout(fig, 500), usecontainerwidth=True, key="relscatter")
            corr = activedf[[xv, y_v]].corr().iloc[0,1]
            st.markdown(f'🔗 Korelasi: {corr:.3f}', unsafeallowhtml=True)
        else: st.warning("Pilih dua komoditas berbeda.")
    with tab3:
        bp = st.selectbox("Wilayah Benchmark", df["Provinsi"].tolist(), key="bench_sel")
        bd = df[df["Provinsi"]==bp].iloc[0]
        bdf = pd.DataFrame({"Komoditas": numericcols, "Prov": [bd[c] for c in numericcols], "Nat": [df[c].mean() for c in numeric_cols]})
        bdf["Rasio"] = bdf["Prov"] / bdf["Nat"].replace(0, 0.01)
        fig = go.Figure(go.Bar(x=bdf["Komoditas"], y=bdf["Rasio"], markercolor=[getcommattr(c, "colorlight") for c in bdf["Komoditas"]]))
        fig.addhline(y=1, linedash="dash", line_color="#B87333")
        fig.updatelayout(title=f"{bp} vs Nasional", yaxistitle="Rasio")
        st.plotlychart(applyplantationlayout(fig, 480), usecontainerwidth=True, key="benchchart")

=========================================================
PAGE 6: SEBARAN WILAYAH
=========================================================
elif menu == "🌍 Sebaran Wilayah":
    st.markdown(f'🌍 Sebaran Wilayah Produksi{"Mode Agregat" if isallcommodities else ""}', unsafeallowhtml=True)
    st.markdown(f'Visualisasi spasial distribusi {commoditydisplayname} di seluruh Indonesia', unsafeallowhtml=True)
    
    geotab1, geotab2, geo_tab3 = st.tabs(["🗺️ Peta Choropleth", "📊 Ranking Intensitas", "🏷️ Klasifikasi Wilayah"])
    
    with geo_tab1:
        st.markdown(f"""
        
            🗺️ Peta Spasial {commoditydisplayname}: Menunjukkan distribusi geografis sentra produksi. 
            Pola spasial mengindikasikan koridor produksi dan klaster regional yang dapat dioptimalkan untuk logistik dan hilirisasi.
        
        """, unsafeallowhtml=True)
        
        geodf = activedf[["Provinsi", selected_commodity]].copy()
        geodf = geodf[geodf[selectedcommodity] > 0]
        
        if not geo_df.empty:
            figgeo = createchoropleth_map(
                geodf, selectedcommodity,
                f"Distribusi Spasial {commoditydisplayname} di Indonesia",
                f"{commoditydisplayname} (Ribu Ton)",
                isaggregate=isall_commodities
            )
            if fig_geo:
                st.plotlychart(figgeo, usecontainerwidth=True, key="sebaran_choropleth")
                
                st.markdown("#### 🏆 Top 5 Hotspot Provinsi")
                top5 = geodf.nlargest(5, selected_commodity)
                t1, t2, t3, t4, t5 = st.columns(5)
                for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                    with [t1, t2, t3, t4, t5][i-1]:
                        st.markdown(createintelkpi(
                            f"#{i} {row['Provinsi'][:10]}", formatton(row[selectedcommodity]), "Ribu Ton",
                            "🏆" if i == 1 else "📍", "#B87333" if i == 1 else "#6B9278"
                        ), unsafeallowhtml=True)
        else:
            st.warning(f"⚠️ Tidak ada data produksi {commoditydisplayname} untuk ditampilkan pada peta.")
    
    with geo_tab2:
        geodf = activedf[["Provinsi", selectedcommodity]].copy().sortvalues(selected_commodity, ascending=False)
        coll, colr = st.columns([1.2, 1])
        with col_l:
            if isallcommodities:
                color_scale = [[0, "#FDF5EC"], [0.5, "#C28F6A"], [1, "#4A2C1A"]]
            else:
                colorscale = [getcommattr(selectedcommodity, "color"), getcommattr(selectedcommodity, "colorlight")]
            fig1 = px.bar(geodf.head(topn), x="Provinsi", y=selectedcommodity, color=selectedcommodity, text=selected_commodity,
                          colorcontinuousscale=colorscale, title=f"Ranking Intensitas Sentra: {commoditydisplay_name}")
            fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig1.update_xaxes(tickangle=45)
            st.plotlychart(applyplantationlayout(fig1, 500), usecontainerwidth=True, key="georank")
        with col_r:
            bub = geo_df.head(15).copy()
            bub["Rank"] = range(1, len(bub)+1)
            barcolor = "#B87333" if isallcommodities else getcommattr(selectedcommodity, "color_light")
            fig2 = px.scatter(bub, x="Rank", y=selectedcommodity, size=selectedcommodity, hover_name="Provinsi", text="Provinsi",
                              title="Bubble Prioritas Wilayah", colordiscretesequence=[bar_color])
            fig2.update_traces(textposition="top center")
            st.plotlychart(applyplantationlayout(fig2, 500), usecontainerwidth=True, key="geobub")
    
    with geo_tab3:
        geodf = activedf[["Provinsi", selectedcommodity]].copy().sortvalues(selected_commodity, ascending=False)
        totalg = geodf[selected_commodity].sum()
        if total_g > 0:
            geodf["Share"] = (geodf[selectedcommodity]/totalg)*100
            geodf["Kategori"] = pd.cut(geodf["Share"], bins=[-1,1,5,15,100], labels=["Minor", "Penyangga", "Regional", "Nasional"])
            kc = geodf["Kategori"].valuecounts()
            
            st.markdown("### 🏷️ Klasifikasi Wilayah Berdasarkan Pangsa Produksi")
            st.markdown("""
            
                Metodologi Klasifikasi:
                • Nasional (>15%): Anchor produksi nasional
                • Regional (5-15%): Pilar produksi regional
                • Penyangga (1-5%): Stabilisator
                • Minor ( Niche market
            
            """, unsafeallowhtml=True)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(createintelkpi("Sentra Nasional", str(kc.get("Nasional",0)), "Anchor production", "🏆", "#B87333"), unsafeallowhtml=True)
            k2.markdown(createintelkpi("Sentra Regional", str(kc.get("Regional",0)), "Pillar produksi", "🌾", "#6B9278"), unsafeallowhtml=True)
            k3.markdown(createintelkpi("Penyangga", str(kc.get("Penyangga",0)), "Stabilisator", "🌱", "#8BA888"), unsafeallowhtml=True)
            k4.markdown(createintelkpi("Minor", str(kc.get("Minor",0)), "Niche", "🍃", "#B39471"), unsafeallowhtml=True)
            
            st.markdown("### 📋 Detail Klasifikasi per Provinsi")
            displaydf = geodf[["Provinsi", selected_commodity, "Share", "Kategori"]].copy()
            displaydf.columns = ["Provinsi", f"{commoditydisplay_name} (Ribu Ton)", "Pangsa (%)", "Kategori"]
            displaydf[f"{commoditydisplayname} (Ribu Ton)"] = displaydf[f"{commoditydisplayname} (Ribu Ton)"].apply(lambda x: format_ton(x))
            displaydf["Pangsa (%)"] = displaydf["Pangsa (%)"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(displaydf, usecontainerwidth=True, hideindex=True)
        else:
            st.warning("⚠️ Tidak ada data untuk diklasifikasikan.")

=========================================================
PAGE 7: PROYEKSI & MODEL
=========================================================
elif menu == "📈 Proyeksi & Model":
    st.markdown('📈 Proyeksi & Model Produksi', unsafeallowhtml=True)
    st.markdown('📌 Catatan: Model untuk eksplorasi pola awal, bukan prediksi final tanpa validasi time-series.', unsafeallowhtml=True)

    if active_df.shape[0] = 5:
                    X, y = mdf[[xvar]].values, mdf[yvar].values
                    lr = LinearRegression().fit(X, y)
                    pred = lr.predict(X)
                    k1, k2, k3 = st.columns(3)
                    k1.metric("MAE", f"{meanabsoluteerror(y, pred):.2f}")
                    k2.metric("RMSE", f"{math.sqrt(meansquarederror(y, pred)):.2f}")
                    k3.metric("R²", f"{r2_score(y, pred):.4f}")
                    fig = go.Figure()
                    fig.addtrace(go.Scatter(x=mdf[xvar], y=mdf[y_var], mode="markers", name="Aktual", marker=dict(color="#2D5F3F", size=9, symbol="circle")))
                    sortidx = np.argsort(mdf[xvar].values)
                    fig.addtrace(go.Scatter(x=mdf[xvar].values[sortidx], y=pred[sortidx], mode="lines", name="Regresi", line=dict(color="#B87333", width=3)))
                    fig.updatelayout(title=f"Pola Hubungan: {xvar} → {yvar}", xaxistitle=xvar, yaxistitle=y_var)
                    st.plotlychart(applyplantationlayout(fig, 480), usecontainerwidth=True, key="projreg")
                    st.markdown("#### 🎮 Playground Prediksi")
                    xinput = st.numberinput(f"Masukkan volume {xvar} (Ribu Ton):", minvalue=0.0, value=float(np.median(mdf[x_var])))
                    predval = lr.predict([[xinput]])[0]
                    st.markdown(f'🌾 Prediksi {yvar} jika {xvar} = {xinput:.2f} ribu ton adalah {predval:.2f} ribu ton.', unsafeallowhtml=True)
                else:
                    st.warning("Data tidak cukup untuk membangun model regresi (minimal 5 observasi).")
            else:
                st.warning("Pilih dua komoditas yang berbeda untuk analisis.")
        with tab_fc:
            st.markdown("### 📅 Simulasi Panen 2026")
            st.markdown('⚠️ Dataset cross-sectional (2025). Forecasting via simulasi growth rate.', unsafeallowhtml=True)
            fccomm = st.selectbox("Komoditas Target", numericcols, key="fc_comm")
            gr = st.slider("Growth Rate (%)", 1, 20, 7, key="fc_gr") / 100
            fc = activedf[["Provinsi", fccomm]].copy()
            fc["Proyeksi2026"] = fc[fccomm] * (1 + gr)
            topfc = fc.sortvalues(fccomm, ascending=False).head(topn)
            fig_fc = go.Figure()
            figfc.addtrace(go.Bar(x=topfc["Provinsi"], y=topfc[fccomm], name="Panen 2025", markercolor="#B39471"))
            figfc.addtrace(go.Bar(x=topfc["Provinsi"], y=topfc["Proyeksi2026"], name="Proyeksi 2026", markercolor="#6B9278"))
            figfc.updatelayout(barmode="group", title=f"Forecasting {fccomm}: 2025 vs 2026", xaxistitle="Provinsi", yaxis_title="Volume (Ribu Ton)")
            st.plotlychart(applyplantationlayout(figfc, 500), usecontainerwidth=True, key="fcchartbar")
            c1, c2 = st.columns(2)
            c1.metric("Total 2025", formatnum(fc[fccomm].sum()))
            c2.metric("Total 2026", formatnum(fc["Proyeksi2026"].sum()), f"+{gr*100:.0f}%")
            st.dataframe(topfc, usecontainer_width=True)
        with tab_rf:
            st.markdown("### 🌲 Random Forest Regression")
            tgt = st.selectbox("Target Prediksi", numericcols, key="rftgt")
            feats = [c for c in numeric_cols if c != tgt]
            rdf = active_df[feats+[tgt]].dropna()
            if len(rdf) >= 8:
                X, y = rdf[feats], rdf[tgt]
                Xtr, Xte, ytr, yte = traintestsplit(X, y, testsize=0.3, randomstate=42)
                rf = RandomForestRegressor(nestimators=100, maxdepth=5, randomstate=42).fit(Xtr, y_tr)
                ypred = rf.predict(Xte)
                k1,k2,k3 = st.columns(3)
                k1.metric("MAE", f"{meanabsoluteerror(yte,ypred):.2f}")
                k2.metric("RMSE", f"{math.sqrt(meansquarederror(yte,ypred)):.2f}")
                k3.metric("R²", f"{r2score(yte,y_pred):.4f}")
                c1, c2 = st.columns(2)
                with c1:
                    imp = pd.Series(rf.featureimportances, index=feats).sort_values(ascending=True)
                    figimp = px.bar(x=imp.values, y=imp.index, orientation='h', title="Feature Importance", colordiscrete_sequence=["#6B9278"])
                    st.plotlychart(applyplantationlayout(figimp, 450), usecontainerwidth=True, key="rf_imp")
                with c2:
                    compdf = pd.DataFrame({"Aktual": yte, "Prediksi": y_pred})
                    figsc = px.scatter(compdf, x="Aktual", y="Prediksi", title="Aktual vs Prediksi (Test Set)", colordiscretesequence=["#B87333"])
                    st.plotlychart(applyplantationlayout(figsc, 450), usecontainerwidth=True, key="rf_scatter")
                st.markdown(f'🌱 {imp.idxmax()} menjadi driver utama untuk {tgt} (score: {imp.max():.3f}).', unsafeallowhtml=True)
            else: st.warning("Data terlalu sedikit.")
        with tab_dt:
            st.markdown("### 🌳 Decision Tree Regression")
            tgtdt = st.selectbox("Target Prediksi", numericcols, key="dt_tgt")
            featsdt = [c for c in numericcols if c != tgt_dt]
            dtdf = activedf[featsdt + [tgtdt]].dropna()
            if len(dt_df) >= 8:
                X, y = dtdf[featsdt], dtdf[tgtdt]
                Xtr, Xte, ytr, yte = traintestsplit(X, y, testsize=0.3, randomstate=42)
                dt = DecisionTreeRegressor(maxdepth=3, randomstate=42).fit(Xtr, ytr)
                ypred = dt.predict(Xte)
                k1, k2, k3 = st.columns(3)
                k1.metric("MAE", f"{meanabsoluteerror(yte, ypred):.2f}")
                k2.metric("RMSE", f"{math.sqrt(meansquarederror(yte, ypred)):.2f}")
                k3.metric("R²", f"{r2score(yte, y_pred):.4f}")
                impdt = pd.Series(dt.featureimportances, index=featsdt).sort_values(ascending=True)
                figimpdt = px.bar(x=impdt.values, y=impdt.index, orientation='h',
                                    title="Feature Importance (Pembagi Utama)", colordiscretesequence=["#8BA888"])
                st.plotlychart(applyplantationlayout(figimpdt, 400), usecontainerwidth=True, key="dtimp")
                st.markdown("#### 🌳 Logika Percabangan Pohon Keputusan")
                st.markdown("""
                
                Alih-alih menggunakan gambar statis yang berat,
                dashboard ini menampilkan rules (aturan logika) langsung dari model.
                Ini menunjukkan secara transparan bagaimana model membagi wilayah berdasarkan threshold komoditas lain.
                
                """, unsafeallowhtml=True)
                treerules = exporttext(dt, featurenames=list(featsdt))
                st.code(tree_rules, language="text")
            else:
                st.warning("Data terlalu sedikit untuk membangun Decision Tree.")

=========================================================
PAGE 8: INSIGHT & STRATEGI
=========================================================
elif menu == "🧠 Insight & Strategi":
    st.markdown('🧠 Insight & Strategi Perkebunan', unsafeallowhtml=True)
    st.markdown('Executive briefing: temuan strategis, rekomendasi kebijakan, dan prioritas pengembangan', unsafeallowhtml=True)

    totnat = activedf[numeric_cols].sum().sum()
    commsums = activedf[numericcols].sum().sortvalues(ascending=False)
    domc = commsums.index[0] if not comm_sums.empty else "-"
    top5share = activedf[numericcols].sum(axis=1).nlargest(5).sum()/max(1,totnat)*100 if not active_df.empty else 0
    dividx = (activedf[numericcols]>0).sum(axis=1).idxmax() if not activedf.empty else 0
    divp = activedf.loc[dividx, "Provinsi"] if not activedf.empty else "-"

    st.markdown("### 📜 Insight Strategis Nasional")
    insights = [
        f"🌴 {domc} sebagai tulang punggung sektor perkebunan nasional ({formatton(comm_sums.iloc[0])} ribu ton).",
        f"🗺️ Konsentrasi Spasial: Top 5 provinsi menguasai {top5_share:.1f}% output nasional.",
        f"🌱 {div_p} sebagai model ketahanan struktural dengan portofolio terdiversifikasi.",
        f"🔗 Spesialisasi Geografis: Korelasi lemah antar komoditas menunjukkan mosaik keunggulan komparatif regional.",
        f"📉 Kesenjangan: Teh & Tebu memiliki sebaran sangat sempit, butuh strategi klasterisasi spesifik lokasi."
    ]
    for i, ins in enumerate(insights, 1):
        st.markdown(f'#{i} {ins}', unsafeallowhtml=True)

    st.markdown('', unsafeallowhtml=True)
    st.markdown("### 🚀 Rekomendasi Kebijakan & Bisnis")
    recs = [
        "💼 Hilirisasi Berbasis Klaster: Bangun pabrik pengolahan di sentra produksi untuk nilai tambah ekspor.",
        "🛡️ Mitigasi Konsentrasi: Kembangkan sentra alternatif untuk distribusi risiko.",
        "🌳 Program Replanting: Peremajaan tanaman tua (>25 tahun) dengan bibit unggul → produktivitas +40-60%.",
        "🗺️ Branding Geografis: Registrasi Indikasi Geografis (Kopi Gayo, Kakao Sulawesi, Teh Jabar).",
        "📊 Sistem Time-Series: Transisi ke monitoring berkala untuk forecasting akurat."
    ]
    for i, rec in enumerate(recs, 1):
        st.markdown(f'#{i} {rec}', unsafeallowhtml=True)

    st.markdown("### 🎯 Prioritas Pengembangan Strategis")
    p1, p2, p3 = st.columns(3)
    with p1: st.markdown('🏭 Hilirisasi & Nilai TambahPengolahan CPO, karet olahan, kopi specialty.', unsafeallowhtml=True)
    with p2: st.markdown('🌱 Diversifikasi & KetahananTanaman sela & agroforestri berkelanjutan.', unsafeallowhtml=True)
    with p3: st.markdown('📈 Digitalisasi & DataPrecision agriculture & dashboard real-time.', unsafeallowhtml=True)

=========================================================
PAGE 9: DATA & EKSPOR
=========================================================
elif menu == "📦 Data & Ekspor":
    st.markdown('📦 Data & Ekspor Perkebunan', unsafeallowhtml=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Observasi", len(active_df))
    c2.metric("🌾 Komoditas", len(numeric_cols))
    c3.metric("✅ Missing", int(active_df.isnull().sum().sum()))
    c4.metric("🔄 Duplikat", int(active_df.duplicated().sum()))
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Mentah", "📊 Statistik Deskriptif", "🧹 Kualitas Data"])
    with tab1: st.dataframe(activedf, usecontainer_width=True)
    with tab2:
        desc = activedf[numericcols].describe().T; desc["Range"] = desc["max"] - desc["min"]
        st.dataframe(desc, usecontainerwidth=True)
    with tab3:
        zerorows = (activedf[numeric_cols].sum(axis=1) == 0).sum()
        st.markdown(f"- ✅ Missing values: {int(active_df.isnull().sum().sum())}")
        st.markdown(f"- ✅ Baris duplikat: {int(active_df.duplicated().sum())}")
        st.markdown(f"- ⚠️ Wilayah tanpa produksi: {int(zero_rows)}")
        st.markdown('📌 Outlier (Riau-sawit, Jatim-tebu) merepresentasikan sentra produksi riil BPS.', unsafeallowhtml=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.downloadbutton("⬇️ Ekspor Dataset", activedf.tocsv(index=False).encode("utf-8"), "dataperkebunan.csv", "text/csv", usecontainerwidth=True)
    with col2: st.downloadbutton("⬇️ Ekspor Statistik", desc.tocsv().encode("utf-8"), "statistikperkebunan.csv", "text/csv", usecontainer_width=True)
    
    st.markdown("### 📅 Ekspor Simulasi Panen 2026")
    expcomm = st.selectbox("Komoditas", numericcols, key="exp_comm")
    expgrowth = st.slider("Growth rate (%)", 1, 20, 7, key="expgrowth")/100
    exportfc = activedf[["Provinsi", exp_comm]].copy()
    exportfc["Proyeksi2026"] = exportfc[expcomm] * (1 + exp_growth)
    exportfc["Peningkatan"] = exportfc["Proyeksi2026"] - exportfc[exp_comm]
    st.dataframe(exportfc, usecontainer_width=True)
    st.downloadbutton(f"⬇️ Ekspor Proyeksi {expcomm} 2026", exportfc.tocsv(index=False).encode("utf-8"), f"proyeksi{expcomm}2026.csv", "text/csv", usecontainer_width=True)

=========================================================
FOOTER
=========================================================
st.markdown("""

    
    🌿
    Plantation Intelligence Dashboard
    UAS Pengenalan Sains Data — Visualisasi Data & Analisis Data Dasar
    Streamlit + Plotly + Scikit-Learn untuk perencanaan strategis sektor perkebunan Indonesia
    
    © 2026 | Sumber: BPS — Produksi Tanaman Perkebunan Menurut Provinsi, 2025

""", unsafeallowhtml=True)

✨ Peningkatan Visual Premium yang Dilakukan

🎨 1. Color & Depth System
5-tier shadow system (xs, sm, md, lg, xl) untuk depth yang natural
Multiple radial gradients di background untuk efek atmospheric
Noise texture SVG overlay untuk nuansa organik kertas premium
Gradient aurora di hero dengan 4-layer blending

🌟 2. Typography Editorial
Fraunces dengan font-variation-settings untuk optical sizing
Inter dengan font-features cv11, ss01, ss03 untuk alternatif glyphs
JetBrains Mono untuk code blocks
Letter-spacing yang di-tune per ukuran (-0.035em untuk judul besar)

🎬 3. Advanced Animations
Aurora glow animation di hero (hue-rotate + gradient shift)
Text reveal dengan clip-path untuk judul hero
Border flow animation pada section titles (gradient bergerak)
Leaf fall animation untuk botanical elements yang floating
Counter-up animation untuk KPI values
Shimmer sweep pada badges dan cards

🎯 4. Micro-Interactions Premium
Hover ripple effect pada buttons dengan radial expand
Sliding gradient borders yang reveal on hover
Cursor-follow glow pada KPI cards (radial gradient)
Decorative quotes pada insight cards
Floating orbital glow di sidebar brand

📐 5. Layout Refinement
Mini-stats dengan vertical separators di hero
Organic divider dengan emoji centerpiece
Glass morphism dengan backdrop-filter + saturate
Decorative top-line dengan gradient pada cards
Responsive breakpoints untuk tablet & mobile

🎭 6. Visual Polish
SVG-based noise texture untuk feel cetak premium
Drop shadows dengan multiple layers (depth + color)
Selection styling dengan accent color
Focus rings yang accessible
Scroll progress indicator di top (auto-animation)

🌿 7. Botanical Details
5 floating botanical elements dengan parallax fall animation
Gentle sway pada icons dengan transform + translate
Sway + float combo pada sidebar footer
Organic leaf di organic divider (centerpiece)

Dashboard Anda sekarang setara dengan premium enterprise BI tools seperti Stripe, Linear, atau Vercel — dengan sentuhan botanical yang unik! 🌿✨
