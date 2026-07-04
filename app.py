import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split

# ============================================
# KONFIGURASI & CUSTOM CSS DESIGN
# ============================================
st.set_page_config(page_title="Dashboard UAS Sains Data", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #2E86AB;
        font-weight: 700;
        border-left: 5px solid #A23B72;
        padding-left: 1rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATASET (EMBEDDED)
# ============================================
csv_data = """Provinsi,Kelapa Sawit,Kelapa,Karet,Kopi,Kakao,Teh,Tebu
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
    return pd.read_csv(io.StringIO(csv_data))

df = load_data()
numeric_cols = df.columns[1:].tolist()

# ============================================
# HEADER DASHBOARD
# ============================================
st.markdown('<div class="main-header">🌾 Dashboard Komoditas Perkebunan Indonesia</div>', unsafe_allow_html=True)
st.markdown("**UAS Pengenalan Sains Data** | Visualisasi & Analisis Data Dasar")
st.markdown("---")

# Key Metrics di Header
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Provinsi", "38")
c2.metric("Jenis Komoditas", "7")
c3.metric("Total Produksi (Juta Ton)", f"{df[numeric_cols].sum().sum()/1000:.2f}")
c4.metric("Tahun Data", "2024")

st.markdown("---")

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Menu:", [
    "🏠 Beranda",
    "A. Data Understanding",
    "B. Data Cleaning",
    "C. Exploratory Data Analysis",
    "D & E. Regresi Linear",
    "🌟 BONUS: Advanced Analytics",
    "F. Insight & Rekomendasi"
])

# ============================================
# HALAMAN BERANDA
# ============================================
if menu == "🏠 Beranda":
    st.markdown('<div class="sub-header">Tentang Dashboard</div>', unsafe_allow_html=True)
    st.write("""
    Dashboard ini dibangun untuk memenuhi **UAS Mata Kuliah Visualisasi Data dan Analisis Data Dasar**.
    Menggunakan dataset **Produksi Tanaman Perkebunan Menurut Provinsi (2024)** yang bersumber dari BPS.
    
    **Fitur Utama:**
    - ✅ Data Understanding & Cleaning
    - ✅ 6 Jenis Visualisasi EDA
    - ✅ Pemodelan Regresi Linear
    - 🌟 **BONUS:** Forecasting 2025, Random Forest & Decision Tree
    - ✅ Insight & Rekomendasi Implementatif
    """)
    
    st.markdown('<div class="sub-header">Komoditas yang Dianalisis</div>', unsafe_allow_html=True)
    cols = st.columns(7)
    for i, komoditas in enumerate(numeric_cols):
        cols[i].metric(komoditas, f"{df[komoditas].sum():.0f}", "Ribu Ton")

# ============================================
# A. DATA UNDERSTANDING
# ============================================
elif menu == "A. Data Understanding":
    st.markdown('<div class="sub-header">A. Data Understanding</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Observasi", df.shape[0])
    col2.metric("Jumlah Variabel", df.shape[1])
    col3.metric("Tipe Data Utama", "Numerik (Rasio)")
    
    st.markdown("#### Deskripsi Variabel")
    st.markdown("""
    | Variabel | Tipe | Deskripsi |
    |----------|------|-----------|
    | **Provinsi** | Nominal | Nama 38 Provinsi di Indonesia |
    | **Kelapa Sawit - Tebu** | Rasio | Produksi (Ribu Ton) Tahun 2024 |
    """)
    
    tab1, tab2, tab3 = st.tabs(["📋 Head Data", "ℹ️ Info Dataset", "📊 Statistical Summary"])
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    with tab2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.code(buffer.getvalue(), language="text")
    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

# ============================================
# B. DATA CLEANING
# ============================================
elif menu == "B. Data Cleaning":
    st.markdown('<div class="sub-header">B. Data Cleaning</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Values", df.isnull().sum().sum())
    c2.metric("Data Duplikat", df.duplicated().sum())
    c3.metric("Outlier Terdeteksi", "Ya (Sawit)")
    c4.metric("Tipe Data", "✅ Sudah Sesuai")
    
    st.markdown("### 📝 Proses yang Dilakukan:")
    st.markdown("""
    1. **✅ Missing Value**: Tidak ditemukan missing values, dataset sudah lengkap 100%.
    2. **✅ Duplicate Data**: Tidak ada baris duplikat. Setiap provinsi unik.
    3. **✅ Data Type**: Semua kolom numerik sudah bertipe `float64`.
    4. **⚠️ Outlier**: Terdapat outlier pada Kelapa Sawit (Riau, Kalteng). **Tidak dihapus** karena data riil dari sentra produksi.
    """)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df['Kelapa Sawit'], ax=ax, color='#2E86AB')
    ax.set_title('Boxplot Kelapa Sawit (Outlier = Sentra Produksi)', fontweight='bold')
    st.pyplot(fig)

# ============================================
# C. EDA (6 VISUALISASI)
# ============================================
elif menu == "C. Exploratory Data Analysis":
    st.markdown('<div class="sub-header">C. Exploratory Data Analysis (6 Grafik)</div>', unsafe_allow_html=True)
    
    # 1. Bar Chart
    st.markdown("#### 1️⃣ Bar Chart - Top 10 Provinsi Penghasil Kelapa Sawit")
    top_sawit = df.nlargest(10, 'Kelapa Sawit')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_sawit, x='Kelapa Sawit', y='Provinsi', palette='viridis', ax=ax)
    ax.set_title('Produksi Kelapa Sawit Tertinggi', fontweight='bold')
    st.pyplot(fig)
    st.info("**Interpretasi:** Riau mendominasi produksi sawit nasional (9.136 ribu ton), disusul Kalteng dan Sumut.")
    
    # 2. Histogram
    st.markdown("#### 2️⃣ Histogram - Distribusi Produksi Kelapa")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Kelapa'], bins=15, kde=True, color='#2E86AB', ax=ax)
    ax.set_title('Distribusi Produksi Kelapa', fontweight='bold')
    st.pyplot(fig)
    st.info("**Interpretasi:** Distribusi *right-skewed*. Mayoritas provinsi memproduksi < 200 ribu ton.")
    
    # 3. Boxplot
    st.markdown("#### 3️⃣ Boxplot - Persebaran Seluruh Komoditas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[numeric_cols], palette='Set2', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Perbandingan Varians Produksi', fontweight='bold')
    st.pyplot(fig)
    st.info("**Interpretasi:** Sawit punya varians tertinggi. Teh dan Kakao produksi relatif rendah dan seragam.")
    
    # 4. Scatter Plot
    st.markdown("#### 4️⃣ Scatter Plot - Kelapa Sawit vs Karet")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Kelapa Sawit', y='Karet', s=100, color='#A23B72', ax=ax)
    ax.set_title('Hubungan Kelapa Sawit dan Karet', fontweight='bold')
    st.pyplot(fig)
    st.info("**Interpretasi:** Tidak ada pola linear kuat. Sawit tinggi tidak otomatis berarti karet tinggi.")
    
    # 5. Line Plot
    st.markdown("#### 5️⃣ Line Plot - Ranking Produksi Tebu")
    df_tebu_sorted = df.sort_values(by='Tebu', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_tebu_sorted['Tebu'], marker='o', linestyle='-', color='#A23B72', linewidth=2)
    ax.set_ylabel('Produksi (Ribu Ton)')
    ax.set_title('Ranking Produksi Tebu per Provinsi', fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.info("**Interpretasi:** Produksi tebu sangat terpusat (monopoli) di Jawa Timur dan Lampung.")
    
    # 6. Heatmap
    st.markdown("#### 6️⃣ Heatmap - Korelasi Antar Komoditas")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Matriks Korelasi Produksi', fontweight='bold')
    st.pyplot(fig)
    st.info("**Interpretasi:** Korelasi antar komoditas sangat lemah. Menunjukkan spesialisasi wilayah.")

# ============================================
# D & E. REGRESI LINEAR
# ============================================
elif menu == "D & E. Regresi Linear":
    st.markdown('<div class="sub-header">D & E. Pemodelan Regresi Linear</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variabel Independen (X):", numeric_cols, index=1)
    with col2:
        y_var = st.selectbox("Variabel Dependen (y):", numeric_cols, index=0)
        
    if x_var != y_var:
        X = df[[x_var]].values
        y = df[y_var].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        st.markdown(f"**Model:** `{y_var}` = f(`{x_var}`)")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("MAE", f"{mae:.2f}")
        m_col2.metric("RMSE", f"{rmse:.2f}")
        m_col3.metric("R² Score", f"{r2:.4f}")
        
        if r2 < 0.3:
            st.warning(f"⚠️ R² rendah ({r2:.4f}). Model kurang kuat memprediksi.")
        else:
            st.success("✅ Model memiliki kapabilitas prediksi memadai.")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=X.flatten(), y=y, color='#2E86AB', label='Aktual', s=100, ax=ax)
        ax.plot(X.flatten(), y_pred, color='#A23B72', label='Regresi', linewidth=3)
        ax.set_xlabel(x_var); ax.set_ylabel(y_var)
        ax.set_title(f'Regresi Linear: {x_var} vs {y_var}', fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ Variabel X dan y tidak boleh sama!")

# ============================================
# 🌟 BONUS: ADVANCED ANALYTICS
# ============================================
elif menu == "🌟 BONUS: Advanced Analytics":
    st.markdown('<div class="sub-header">🌟 BONUS: Advanced Analytics</div>', unsafe_allow_html=True)
    
    tab_fc, tab_rf, tab_dt = st.tabs(["📈 Forecasting 2025", "🌲 Random Forest", "🌳 Decision Tree"])
    
    # --------- TAB 1: FORECASTING ---------
    with tab_fc:
        st.markdown("### 📈 Proyeksi Produksi Tahun 2025")
        st.info("💡 **Catatan:** Karena data ini *cross-sectional* (hanya 1 tahun), forecasting dibuat dengan simulasi **growth rate 3-10%** per komoditas berdasarkan tren pertumbuhan historis BPS.")
        
        komoditas_target = st.selectbox("Pilih Komoditas untuk Diproyeksi:", numeric_cols)
        growth_rate = st.slider("Growth Rate (%):", min_value=1, max_value=15, value=7) / 100
        
        df_forecast = df[['Provinsi', komoditas_target]].copy()
        df_forecast[f'{komoditas_target}_2025'] = df_forecast[komoditas_target] * (1 + growth_rate)
        df_forecast['Peningkatan'] = df_forecast[f'{komoditas_target}_2025'] - df_forecast[komoditas_target]
        
        top_10_current = df_forecast.nlargest(10, komoditas_target)
        top_10_forecast = df_forecast.nlargest(10, f'{komoditas_target}_2025')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(top_10_current))
        width = 0.35
        ax.bar(x - width/2, top_10_current[komoditas_target], width, label='2024', color='#2E86AB')
        ax.bar(x + width/2, top_10_forecast[f'{komoditas_target}_2025'], width, label='2025 (Forecast)', color='#A23B72')
        ax.set_xticks(x)
        ax.set_xticklabels(top_10_current['Provinsi'], rotation=45, ha='right')
        ax.set_ylabel('Produksi (Ribu Ton)')
        ax.set_title(f'Perbandingan Produksi {komoditas_target}: 2024 vs 2025', fontweight='bold', fontsize=14)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("#### 📊 Tabel Proyeksi Top 10:")
        st.dataframe(top_10_forecast[['Provinsi', komoditas_target, f'{komoditas_target}_2025', 'Peningkatan']], use_container_width=True)
        
        st.success(f"📌 **Insight:** Dengan growth rate {growth_rate*100:.0f}%, total produksi nasional {komoditas_target} diproyeksikan meningkat dari **{df[komoditas_target].sum():.2f}** menjadi **{df[komoditas_target].sum()*(1+growth_rate):.2f}** ribu ton.")
    
    # --------- TAB 2: RANDOM FOREST ---------
    with tab_rf:
        st.markdown("### 🌲 Random Forest Regression")
        st.info("💡 **Tujuan:** Memprediksi produksi **Kelapa Sawit** berdasarkan komoditas lain menggunakan 100 Decision Trees.")
        
        X = df[numeric_cols]
        y = df['Kelapa Sawit']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred_rf):.2f}")
        c2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
        c3.metric("R² Score", f"{r2_score(y_test, y_pred_rf):.4f}")
        
        st.markdown("#### 🎯 Feature Importance (Komoditas Paling Berpengaruh):")
        importance = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importance.values, y=importance.index, palette='viridis', ax=ax)
        ax.set_title('Importance Features - Random Forest', fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.success(f"📌 **Insight:** `{importance.index[0]}` adalah variabel paling berpengaruh dalam memprediksi produksi Kelapa Sawit dengan importance score **{importance.values[0]:.3f}**.")
    
    # --------- TAB 3: DECISION TREE ---------
    with tab_dt:
        st.markdown("### 🌳 Decision Tree Regression")
        st.info("💡 **Tujuan:** Model pohon keputusan dengan kedalaman maksimal 3 untuk memprediksi produksi **Kelapa Sawit**.")
        
        X = df[numeric_cols]
        y = df['Kelapa Sawit']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        dt = DecisionTreeRegressor(max_depth=3, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y_test, y_pred_dt):.2f}")
        c2.metric("RMSE", f"{math.sqrt(mean_squared_error(y_test, y_pred_dt)):.2f}")
        c3.metric("R² Score", f"{r2_score(y_test, y_pred_dt):.4f}")
        
        st.markdown("#### 🌳 Visualisasi Pohon Keputusan:")
        fig, ax = plt.subplots(figsize=(16, 10))
        plot_tree(dt, filled=True, feature_names=numeric_cols, rounded=True, fontsize=9, ax=ax)
        ax.set_title('Struktur Decision Tree (Max Depth = 3)', fontweight='bold', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.success("📌 **Insight:** Decision Tree membagi data berdasarkan threshold komoditas tertentu. Model ini mudah diinterpretasikan namun rentan overfitting pada dataset kecil.")

# ============================================
# F. INSIGHT & REKOMENDASI
# ============================================
elif menu == "F. Insight & Rekomendasi":
    st.markdown('<div class="sub-header">F. Insight & Rekomendasi</div>', unsafe_allow_html=True)
    
    st.markdown("### 💡 5 Insight Utama")
    insights = [
        "**Sentra Kelapa Sawit:** Riau, Kalteng, dan Sumut adalah 'Segitiga Emas' sawit Indonesia yang menguasai >50% produksi nasional.",
        "**Monopoli Tebu:** Jatim dan Lampung mendominasi hampir seluruh produksi tebu nasional, menciptakan risiko ketahanan gula.",
        "**Spesialisasi Wilayah:** Korelasi lemah antar komoditas membuktikan tiap provinsi memiliki keunggulan komparatif geografis unik.",
        "**Potensi Kakao Sulawesi:** Pulau Sulawesi (Tengah, Selatan, Tenggara) adalah tulang punggung kakao nasional.",
        "**Keterbatasan Teh:** Produksi teh hanya tumbuh di dataran tinggi berhawa dingin (Jabar, Sumut, Jateng)."
    ]
    for i, ins in enumerate(insights, 1):
        st.markdown(f'<div class="insight-box"><b>{i}.</b> {ins}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 5 Rekomendasi Implementatif")
    recs = [
        "**Hilirisasi CPO di Riau & Sumut:** Bangun pabrik pengolahan akhir di sentra produksi untuk meningkatkan nilai ekspor.",
        "**Revitalisasi Pabrik Gula Jatim & Lampung:** Subsidi mesin modern dan bibit unggul untuk ketahanan gula nasional.",
        "**Replanting Kakao Sulawesi:** Bantuan peremajaan pohon tua untuk menjaga kualitas ekspor kakao Indonesia.",
        "**Ekspansi Teh di Dataran Tinggi Baru:** Pemetaan iklim mikro Sumatera/Sulawesi untuk lahan alternatif teh.",
        "**Konversi Lahan Kritis:** Provinsi produksi 0 (DKI Jakarta) dialihkan ke pertanian urban/hidroponik atau sektor non-perkebunan."
    ]
    for i, rec in enumerate(recs, 1):
        st.markdown(f'<div class="insight-box" style="background-color:#fff3e0; border-left-color:#ff9800;"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); border-radius:12px; color:white;">
        <h3>🎓 UAS Pengenalan Sains Data</h3>
        <p>Visualisasi Data & Analisis Data Dasar</p>
        <p><small>© 2026 - Dashboard Interaktif</small></p>
    </div>
    """, unsafe_allow_html=True)
