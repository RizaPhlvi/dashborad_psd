import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fallback aman untuk server headless
import matplotlib.pyplot as plt
import seaborn as sns
import io
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Dashboard UAS Sains Data", layout="wide")

# Embed Dataset langsung ke dalam kode
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
    df = pd.read_csv(io.StringIO(csv_data))
    return df

df = load_data()
numeric_cols = df.columns[1:]

st.title("📊 Dashboard Analisis Komoditas Perkebunan Indonesia (2024)")
st.markdown("**UAS Pengenalan Sains Data - Visualisasi Data dan Analisis Data Dasar**")
st.markdown("---")

menu = st.sidebar.radio("Menu Navigasi UAS", [
    "A. Data Understanding",
    "B. Data Cleaning",
    "C. Exploratory Data Analysis (EDA)",
    "D & E. Analisis Hubungan & Regresi",
    "F. Insight & Rekomendasi"
])

if menu == "A. Data Understanding":
    st.header("A. Data Understanding")
    st.write("Dataset berisi data Produksi Tanaman Perkebunan Menurut Provinsi dan Jenis Tanaman (Ribu Ton) tahun 2024.")
    
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Observasi (Baris)", df.shape[0])
    col2.metric("Jumlah Variabel (Kolom)", df.shape[1])
    
    st.subheader("Deskripsi Variabel")
    st.markdown("""
    - **Provinsi**: Nama Provinsi di Indonesia (Nominal/String)
    - **Kelapa Sawit s/d Tebu**: Produksi masing-masing komoditas dalam satuan Ribu Ton (Rasio/Numerik)
    """)
    
    st.subheader("Head Dataset")
    st.dataframe(df.head())
    
    st.subheader("Info Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue(), language="text")
    
    st.subheader("Statistical Describe")
    st.dataframe(df.describe())

elif menu == "B. Data Cleaning":
    st.header("B. Data Cleaning")
    
    st.subheader("1. Missing Values")
    missing = df.isnull().sum().sum()
    st.write(f"Total missing values pada dataset: **{missing}**")
    st.success("Aman: Tidak terdapat missing values, sehingga tidak perlu imputasi.")
    
    st.subheader("2. Duplicate Data")
    duplicates = df.duplicated().sum()
    st.write(f"Total baris duplikat: **{duplicates}**")
    st.success("Aman: Tidak ada data duplikat. Dataset sudah unik per provinsi.")
    
    st.subheader("3. Data Type")
    st.dataframe(df.dtypes.reset_index().rename(columns={'index':'Kolom', 0:'Tipe Data'}))
    
    st.subheader("4. Outlier Detection")
    st.write("Visualisasi Boxplot untuk mendeteksi Outlier pada Kelapa Sawit:")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Kelapa Sawit'], ax=ax, color='lightblue')
    st.pyplot(fig)
    st.info("**Keputusan:** Outlier tidak dihapus karena merepresentasikan provinsi sentra produksi utama (seperti Riau & Kalteng) yang secara alamiah memang produksinya masif.")

elif menu == "C. Exploratory Data Analysis (EDA)":
    st.header("C. Exploratory Data Analysis (EDA)")
    
    st.subheader("1. Bar Chart: Top 10 Provinsi Penghasil Kelapa Sawit")
    top_sawit = df.nlargest(10, 'Kelapa Sawit')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_sawit, x='Kelapa Sawit', y='Provinsi', palette='viridis', ax=ax)
    st.pyplot(fig)
    st.info("**Interpretasi:** Riau mendominasi produksi Kelapa Sawit nasional, disusul Kalimantan Tengah dan Sumatera Utara.")
    
    st.subheader("2. Histogram: Distribusi Produksi Kelapa")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Kelapa'], bins=15, kde=True, color='green', ax=ax)
    st.pyplot(fig)
    st.info("**Interpretasi:** Distribusi produksi kelapa *right-skewed*. Mayoritas provinsi memproduksi di bawah 200 ribu ton.")
    
    st.subheader("3. Boxplot: Persebaran Seluruh Komoditas")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[numeric_cols], palette='Set2', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.info("**Interpretasi:** Kelapa Sawit memiliki varians produksi tertinggi. Teh dan Kakao produksinya relatif rendah dan seragam.")
    
    st.subheader("4. Scatter Plot: Hubungan Kelapa Sawit vs Karet")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Kelapa Sawit', y='Karet', s=100, color='orange', ax=ax)
    st.pyplot(fig)
    st.info("**Interpretasi:** Tidak ada pola linear yang kuat. Provinsi penghasil sawit tinggi tidak otomatis menjadi penghasil karet tinggi.")
    
    st.subheader("5. Line Plot: Penurunan Produksi Tebu (Sorted)")
    df_tebu_sorted = df.sort_values(by='Tebu', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_tebu_sorted['Tebu'], marker='o', linestyle='-', color='purple')
    ax.set_ylabel('Produksi (Ribu Ton)')
    st.pyplot(fig)
    st.info("**Interpretasi:** Produksi Tebu sangat terpusat (monopoli) di Jawa Timur dan Lampung. Provinsi lain produksinya mendekati nol.")
    
    st.subheader("6. Heatmap Correlation")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.info("**Interpretasi:** Korelasi antar komoditas sangat lemah (mendekati 0). Menunjukkan adanya spesialisasi wilayah yang kuat (faktor geografis/iklim).")

elif menu == "D & E. Analisis Hubungan & Regresi":
    st.header("D & E. Pemodelan Regresi Linear")
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Pilih Variabel Independen (X):", numeric_cols, index=1)
    with col2:
        y_var = st.selectbox("Pilih Variabel Dependen (y):", numeric_cols, index=0)
        
    if x_var != y_var:
        X = df[[x_var]].values
        y = df[y_var].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        st.markdown(f"**Model:** Memprediksi `{y_var}` berdasarkan `{x_var}`")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("MAE", f"{mae:.2f}")
        m_col2.metric("RMSE", f"{rmse:.2f}")
        m_col3.metric("R² Score", f"{r2:.4f}")
        
        st.markdown("**Interpretasi Model:**")
        if r2 < 0.3:
            st.warning(f"Nilai R² ({r2:.4f}) sangat rendah. Variabel `{x_var}` tidak cukup kuat memprediksi `{y_var}`. Produksi perkebunan lebih dipengaruhi faktor luar seperti luas lahan dan iklim yang tidak ada di dataset ini.")
        else:
            st.success("Model memiliki kapabilitas prediksi yang memadai.")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=X.flatten(), y=y, color='blue', label='Data Aktual', ax=ax)
        ax.plot(X.flatten(), y_pred, color='red', label='Garis Regresi')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        st.pyplot(fig)
    else:
        st.error("Variabel X dan y tidak boleh sama!")

elif menu == "F. Insight & Rekomendasi":
    st.header("F. Insight dan Rekomendasi")
    
    st.subheader("💡 5 Insight Utama")
    st.markdown("""
    1. **Sentra Kelapa Sawit:** Riau, Kalteng, dan Sumut adalah "Segitiga Emas" sawit Indonesia.
    2. **Monopoli Tebu:** Jatim dan Lampung menguasai hampir seluruh produksi tebu nasional.
    3. **Spesialisasi Wilayah:** Lemahnya korelasi antar komoditas membuktikan bahwa tiap provinsi memiliki keunggulan komparatif geografis yang unik.
    4. **Potensi Kakao Sulawesi:** Sulawesi (Tengah, Selatan, Tenggara) adalah tulang punggung kakao nasional.
    5. **Keterbatasan Teh:** Produksi teh hanya hidup di wilayah berhawa dingin (Jabar, Sumut, Jateng).
    """)
    
    st.subheader("🚀 5 Rekomendasi Implementatif")
    st.markdown("""
    1. **Hilirisasi di Riau & Sumut:** Pemerintah harus membangun pabrik pengolahan CPO dan Karet akhir di daerah ini agar nilai ekspor meningkat.
    2. **Revitalisasi Pabrik Gula:** Fokuskan subsidi mesin dan bibit tebu unggul di Jatim dan Lampung untuk ketahanan gula nasional.
    3. **Replanting Kakao Sulawesi:** Berikan bantuan peremajaan pohon kakao tua di Sulawesi untuk mencegah penurunan kualitas ekspor.
    4. **Ekspansi Teh Terukur:** Lakukan pemetaan iklim mikro di dataran tinggi Sumatera/Sulawesi untuk mencari lahan alternatif teh.
    5. **Alih Komoditas Lahan Kritis:** Provinsi dengan produksi nol (seperti DKI Jakarta) sebaiknya dialihkan ke pertanian urban/hidroponik atau sektor non-perkebunan.
    """)
