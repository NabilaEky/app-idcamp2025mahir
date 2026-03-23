import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# STYLE
plt.style.use('default')
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Student Dropout Dashboard", layout="wide")

st.title("🎓 Business Dashboard - Student Dropout Analysis")
st.markdown("### 📊 Monitoring Faktor Penyebab Dropout Mahasiswa")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\Submission\data.csv", sep=';')
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

df = load_data()

# =========================
# VALIDASI KOLOM
# =========================
required_cols = [
    'Status',
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_grade'
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom '{col}' tidak ditemukan! Kolom tersedia: {df.columns}")
        st.stop()

# =========================
# FILTER
# =========================
st.sidebar.header("Filter Data")

status_filter = st.sidebar.multiselect(
    "Pilih Status",
    options=df['Status'].unique(),
    default=df['Status'].unique()
)

df = df[df['Status'].isin(status_filter)]

# =========================
# KPI
# =========================
total = len(df)
dropout = len(df[df['Status'] == 'Dropout'])
graduate = len(df[df['Status'] == 'Graduate'])

dropout_rate = (dropout / total) * 100 if total > 0 else 0

col1, col2, col3 = st.columns(3)

col1.metric("Total Mahasiswa", total)
col2.metric("Dropout", dropout)
col3.metric("Dropout Rate", f"{dropout_rate:.2f}%")

st.markdown("---")

# =========================
# VISUALISASI
# =========================
st.subheader("📊 Visualisasi Data")

# ROW 1
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Distribusi Status")

    fig1, ax1 = plt.subplots()
    colors = ['#4CAF50', '#F44336', '#2196F3']

    df['Status'].value_counts().plot(
        kind='bar',
        ax=ax1,
        color=colors
    )

    ax1.set_ylabel("Jumlah")
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=0)

    for p in ax1.patches:
        ax1.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center',
            va='bottom',
            xytext=(0, 3),
            textcoords='offset points'
        )

    st.pyplot(fig1)

with col2:
    st.markdown("### 2. Pengaruh status pembayaran terhadap dropout")

    fig2, ax2 = plt.subplots()

    sns.countplot(
        data=df,
        x='Tuition_fees_up_to_date',
        hue='Status',
        palette='Set2',
        ax=ax2
    )

    ax2.set_xlabel("Pembayaran")
    ax2.set_ylabel("Jumlah")

    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2, height),
                ha='center',
                va='bottom',
                xytext=(0, 3),
                textcoords='offset points',
                fontsize=9
            )

    st.pyplot(fig2)

st.markdown("---")

# ROW 2
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 3. Pengaruh beasiswa terhadap dropout")

    fig3, ax3 = plt.subplots()

    sns.countplot(
        data=df,
        x='Scholarship_holder',
        hue='Status',
        palette='Set3',
        ax=ax3
    )

    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2, height),
                ha='center',
                va='bottom',
                xytext=(0, 3),
                textcoords='offset points',
                fontsize=9
            )

    st.pyplot(fig3)

with col4:
    st.markdown("### 4. Pengaruh nilai semester 1 terhadap status siswa")

    fig4, ax4 = plt.subplots()

    sns.boxplot(
        data=df,
        x='Status',
        y='Curricular_units_1st_sem_grade',
        palette='pastel',
        ax=ax4
    )

    st.pyplot(fig4)

st.markdown("---")

# ROW 3
col5, col6 = st.columns(2)

with col5:
    st.markdown("### 5. Pengaruh nilai semester 2 terhadap status siswa")

    fig5, ax5 = plt.subplots()

    sns.boxplot(
        data=df,
        x='Status',
        y='Curricular_units_2nd_sem_grade',
        palette='coolwarm',
        ax=ax5
    )

    st.pyplot(fig5)

# =========================
# FEATURE IMPORTANCE
# =========================
with col6:
    st.markdown("### 6. 10 Faktor Utama Dropout")

    try:
        df_model = df.copy()

        df_model['Status'] = df_model['Status'].map({
            'Dropout': 0,
            'Enrolled': 1,
            'Graduate': 2
        })

        X = df_model.select_dtypes(include=['int64', 'float64']).drop(columns=['Status'], errors='ignore')
        y = df_model['Status']

        X = X.fillna(0)

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        feat_importance = pd.Series(model.feature_importances_, index=X.columns)
        feat_importance = feat_importance.sort_values(ascending=False)

        fig6, ax6 = plt.subplots()

        feat_importance.head(10).plot(
            kind='barh',
            ax=ax6,
            color='#FF9800'
        )

        ax6.set_xlabel("Importance Score")
        ax6.set_title("Top 10 Faktor Penyebab Dropout")

        ax6.grid(axis='x', linestyle='--', alpha=0.5)

        for i, v in enumerate(feat_importance.head(10)):
            ax6.text(v, i, f"{v:.2f}", va='center')

        ax6.invert_yaxis()

        st.pyplot(fig6)

    except Exception as e:
        st.error(f"Gagal membuat feature importance: {e}")

st.markdown("---")