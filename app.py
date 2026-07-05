# =========================================================
# PLANTATION PREMIUM DARK THEME CSS - ENHANCED
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ============ ROOT VARIABLES ============ */
:root {
    --bg-deep: #050d08;
    --bg-mid: #0a1a10;
    --bg-surface: #0d2214;
    --accent-primary: #4ade80;
    --accent-secondary: #22c55e;
    --accent-gold: #f59e0b;
    --accent-warm: #d97706;
    --text-primary: #f0fdf4;
    --text-secondary: #bbf7d0;
    --text-muted: #6ee7b7;
    --glass-bg: rgba(13, 34, 20, 0.6);
    --glass-border: rgba(74, 222, 128, 0.15);
}

/* ============ BASE STYLES ============ */
.stApp {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    position: relative;
    overflow-x: hidden;
}

/* ============ ANIMATED BACKGROUND ============ */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(ellipse at 20% 50%, rgba(34, 197, 94, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(245, 158, 11, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 40% 20%, rgba(74, 222, 128, 0.06) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: ambientPulse 8s ease-in-out infinite;
}

@keyframes ambientPulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

/* ============ FLOATING LEAF PARTICLES ============ */
.stApp::after {
    content: '🍃🌿🍂🌱🍃🌿';
    position: fixed;
    top: -100px;
    left: 0;
    width: 100%;
    height: 100%;
    font-size: 2rem;
    opacity: 0.03;
    pointer-events: none;
    z-index: 1;
    animation: floatLeaves 30s linear infinite;
    letter-spacing: 100px;
}

@keyframes floatLeaves {
    0% { transform: translateY(-100px) rotate(0deg); }
    100% { transform: translateY(calc(100vh + 100px)) rotate(360deg); }
}

/* ============ LAYOUT CONTAINER ============ */
.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    position: relative;
    z-index: 2;
}

/* ============ HERO STRIP - ENHANCED ============ */
.hero-strip {
    background: 
        linear-gradient(135deg, rgba(13, 34, 20, 0.95) 0%, rgba(26, 77, 46, 0.9) 50%, rgba(47, 133, 90, 0.85) 100%);
    padding: 2.8rem 3rem;
    border-radius: 24px;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    box-shadow: 
        0 25px 60px rgba(34, 197, 94, 0.2),
        0 0 100px rgba(74, 222, 128, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.hero-strip::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(74, 222, 128, 0.15) 0%, transparent 70%);
    animation: heroGlow 6s ease-in-out infinite;
    pointer-events: none;
}

.hero-strip::after {
    content: '🌿';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 12rem;
    opacity: 0.08;
    filter: blur(2px);
    animation: leafSway 4s ease-in-out infinite;
}

@keyframes heroGlow {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

@keyframes leafSway {
    0%, 100% { transform: translateY(-50%) rotate(-5deg); }
    50% { transform: translateY(-50%) rotate(5deg); }
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.6rem;
    position: relative;
    z-index: 2;
    text-shadow: 0 2px 20px rgba(74, 222, 128, 0.3);
    letter-spacing: -0.02em;
}

.hero-accent {
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, #4ade80, #f59e0b, #4ade80);
    border-radius: 2px;
    margin-bottom: 1rem;
    position: relative;
    z-index: 2;
    animation: accentShimmer 3s ease-in-out infinite;
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
}

@keyframes accentShimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.9);
    line-height: 1.7;
    max-width: 85%;
    margin-bottom: 1.5rem;
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
    padding: 0.5rem 1.1rem;
    border-radius: 999px;
    background: rgba(74, 222, 128, 0.1);
    border: 1px solid rgba(74, 222, 128, 0.3);
    font-size: 0.85rem;
    font-weight: 600;
    color: #a7f3d0;
    box-shadow: 
        0 0 20px rgba(74, 222, 128, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.hero-badge:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 8px 25px rgba(74, 222, 128, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.15);
    background: rgba(74, 222, 128, 0.15);
}

.hero-mini-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,0.15);
    position: relative;
    z-index: 2;
    flex-wrap: wrap;
}

.mini-stat-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.mini-stat-value {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff;
    text-shadow: 0 0 10px rgba(74, 222, 128, 0.3);
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: 
        linear-gradient(180deg, #050d08 0%, #0a1a10 100%);
    border-right: 1px solid rgba(74, 222, 128, 0.1);
    position: relative;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, transparent, rgba(74, 222, 128, 0.3), transparent);
}

.sidebar-block {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.sidebar-block:hover {
    border-color: rgba(74, 222, 128, 0.25);
    box-shadow: 0 12px 40px rgba(34, 197, 94, 0.15);
}

.sidebar-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--accent-primary);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.commodity-brief {
    background: rgba(34, 197, 94, 0.08);
    border-left: 3px solid var(--accent-primary);
    padding: 1rem;
    border-radius: 12px;
    margin-top: 0.6rem;
    backdrop-filter: blur(5px);
}

.active-filter-box {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 12px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #fef3c7;
    margin-top: 0.6rem;
}

/* ============ KPI CARDS - GLASSMORPHISM ============ */
.intel-kpi {
    background: 
        linear-gradient(135deg, rgba(13, 34, 20, 0.8) 0%, rgba(10, 26, 16, 0.9) 100%);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 
        0 10px 40px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.intel-kpi::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.intel-kpi:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 
        0 20px 50px rgba(34, 197, 94, 0.2),
        0 0 60px rgba(74, 222, 128, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    border-color: rgba(74, 222, 128, 0.3);
}

.intel-kpi:hover::before {
    opacity: 1;
}

.kpi-layer1 {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.kpi-layer2 {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 0.4rem;
    text-shadow: 0 2px 10px rgba(74, 222, 128, 0.2);
}

.kpi-layer3 {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ============ SECTION TITLES ============ */
.section-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, var(--accent-primary), var(--accent-gold)) 1;
    display: inline-block;
    position: relative;
    letter-spacing: -0.01em;
}

.section-title::before {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 60px;
    height: 2px;
    background: var(--accent-gold);
    box-shadow: 0 0 10px var(--accent-gold);
}

.section-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
    font-style: italic;
    font-weight: 400;
}

/* ============ INSIGHT CARDS ============ */
.insight-card {
    background: 
        linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(74, 222, 128, 0.04) 100%);
    border-left: 4px solid var(--accent-primary);
    padding: 1.3rem 1.5rem;
    border-radius: 16px;
    color: var(--text-secondary);
    margin: 1rem 0;
    box-shadow: 
        0 8px 24px rgba(34, 197, 94, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    line-height: 1.7;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(74, 222, 128, 0.15);
    transition: all 0.3s ease;
}

.insight-card:hover {
    transform: translateX(5px);
    box-shadow: 
        0 12px 32px rgba(34, 197, 94, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.watchlist-card {
    background: 
        linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(217, 119, 6, 0.04) 100%);
    border-left: 4px solid var(--accent-gold);
    padding: 1.3rem 1.5rem;
    border-radius: 16px;
    color: #fef3c7;
    margin: 1rem 0;
    line-height: 1.7;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(245, 158, 11, 0.2);
    box-shadow: 0 8px 24px rgba(245, 158, 11, 0.1);
}

.rec-card {
    background: 
        linear-gradient(135deg, rgba(132, 204, 22, 0.08) 0%, rgba(163, 230, 53, 0.04) 100%);
    border-left: 4px solid #84cc16;
    padding: 1.3rem 1.5rem;
    border-radius: 16px;
    color: #ecfccb;
    margin: 0.8rem 0;
    line-height: 1.7;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(132, 204, 22, 0.2);
}

.priority-card {
    background: 
        linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(125, 211, 252, 0.04) 100%);
    border-left: 4px solid #38bdf8;
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    color: #e0f2fe;
    margin: 0.6rem 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(56, 189, 248, 0.2);
}

/* ============ TABS ============ */
button[data-baseweb="tab"] {
    border-radius: 12px !important;
    font-weight: 600 !important;
    background: rgba(13, 34, 20, 0.6) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--glass-border) !important;
    padding: 0.8rem 1.4rem !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease !important;
}

button[data-baseweb="tab"]:hover {
    background: rgba(34, 197, 94, 0.1) !important;
    border-color: rgba(74, 222, 128, 0.3) !important;
    transform: translateY(-2px);
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: 
        linear-gradient(135deg, rgba(26, 77, 46, 0.9) 0%, rgba(47, 133, 90, 0.9) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(74, 222, 128, 0.4) !important;
    box-shadow: 
        0 8px 20px rgba(34, 197, 94, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

/* ============ FORM ELEMENTS ============ */
.stSelectbox label, .stSlider label, .stRadio label, .stMultiSelect label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-weight: 600;
    font-size: 0.9rem;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(13, 34, 20, 0.6) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
}

/* ============ METRICS ============ */
div[data-testid="stMetric"] {
    display: none;
}

/* ============ HEADER ============ */
header[data-testid="stHeader"] {
    background: rgba(5, 13, 8, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(74, 222, 128, 0.1);
}

/* ============ DATAFRAMES ============ */
.stDataFrame {
    border-radius: 16px;
    border: 1px solid var(--glass-border) !important;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.stDataFrame > div {
    background: rgba(13, 34, 20, 0.6) !important;
    backdrop-filter: blur(10px);
}

/* ============ BUTTONS ============ */
.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.8), rgba(22, 163, 74, 0.8));
    color: white;
    border: 1px solid rgba(74, 222, 128, 0.3);
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(34, 197, 94, 0.3);
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.9), rgba(22, 163, 74, 0.9));
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(13, 34, 20, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(74, 222, 128, 0.3);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(74, 222, 128, 0.5);
}

/* ============ CHART CONTAINERS ============ */
.js-plotly-plot .plotly {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid var(--glass-border);
}

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
    .hero-title { font-size: 1.8rem; }
    .hero-strip { padding: 2rem 1.5rem; }
    .kpi-layer2 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)
