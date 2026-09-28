import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI-Audit", page_icon="🔎", layout="wide")

st.title("🔎 AI-Audit")
st.caption("نظام ذكي لتحليل مخاطر العمليات المحاسبية واكتشاف العمليات غير المعتادة")

# -----------------------------
# Demo data
# -----------------------------
def demo_data():
    rng = np.random.default_rng(42)
    n = 250
    amounts = rng.lognormal(mean=8.0, sigma=0.75, size=n).round(2)
    # Add some intentionally unusual demo transactions
    amounts[:8] = [250000, 180000, 320000, 210000, 275000, 195000, 400000, 225000]

    types = rng.choice(["Sales", "Purchase", "Expense"], size=n, p=[0.45, 0.35, 0.20])
    departments = rng.choice(["Sales", "Purchasing", "Finance", "HR", "Operations"], size=n)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "date": dates,
        "type": types,
        "amount": amounts,
        "department": departments,
    })

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "transaction": "transaction_id",
        "id": "transaction_id",
        "transactionid": "transaction_id",
        "value": "amount",
        "price": "amount",
        "total": "amount",
        "transaction_type": "type",
        "category": "type",
        "dept": "department",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    required = ["amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("الملف يجب أن يحتوي على عمود amount على الأقل.")

    if "transaction_id" not in df.columns:
        df["transaction_id"] = np.arange(1, len(df) + 1)

    if "date" not in df.columns:
        df["date"] = pd.Timestamp.today().normalize()

    if "type" not in df.columns:
        df["type"] = "Unknown"

    if "department" not in df.columns:
        df["department"] = "Unknown"

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["amount"]).reset_index(drop=True)
    return df

def analyze(df):
    df = df.copy()

    mean = df["amount"].mean()
    std = df["amount"].std(ddof=0)
    unusual_limit = mean + 2 * std if std > 0 else mean

    def risk_level(x):
        if x >= max(10000, mean + std):
            return "High Risk"
        if x >= max(5000, mean):
            return "Medium Risk"
        return "Low Risk"

    df["risk_level"] = df["amount"].apply(risk_level)

    score = np.where(
        df["amount"] >= max(10000, mean + std), 70,
        np.where(df["amount"] >= max(5000, mean), 40, 15)
    )

    df["risk_score"] = score.astype(int)

    # Machine-learning anomaly detection.
    # This detects statistical outliers; it does NOT prove fraud.
    if len(df) >= 20 and df["amount"].nunique() > 1:
        model = IsolationForest(
            n_estimators=150,
            contamination="auto",
            random_state=42
        )
        df["ml_flag"] = model.fit_predict(df[["amount"]])
        df["ml_anomaly"] = np.where(df["ml_flag"] == -1, "Yes", "No")
    else:
        df["ml_anomaly"] = "Not enough data"

    df["unusual_transaction"] = np.where(
        df["amount"] > unusual_limit, "Yes", "No"
    )

    return df, unusual_limit

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("مصدر البيانات")
uploaded = st.sidebar.file_uploader("ارفع Excel أو CSV", type=["xlsx", "csv"])

if uploaded is None:
    df = demo_data()
    st.sidebar.info("يتم استخدام بيانات تجريبية. ارفع ملفك لتحليل بيانات حقيقية.")
else:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        df = normalize_columns(df)
    except Exception as e:
        st.error(f"تعذر قراءة الملف: {e}")
        st.stop()

try:
    df = normalize_columns(df)
except Exception as e:
    st.error(str(e))
    st.stop()

df, unusual_limit = analyze(df)

# -----------------------------
# KPIs
# -----------------------------
total = len(df)
total_amount = df["amount"].sum()
high = int((df["risk_level"] == "High Risk").sum())
medium = int((df["risk_level"] == "Medium Risk").sum())
unusual = int((df["unusual_transaction"] == "Yes").sum())
ml_anomalies = int((df["ml_anomaly"] == "Yes").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("إجمالي العمليات", f"{total:,}")
c2.metric("إجمالي القيمة", f"{total_amount:,.2f}")
c3.metric("High Risk", f"{high:,}")
c4.metric("Medium Risk", f"{medium:,}")
c5.metric("ML Anomalies", f"{ml_anomalies:,}")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🔍 العمليات", "🤖 التحليل الذكي", "📄 التقرير"]
)

with tab1:
    st.subheader("توزيع مستويات المخاطر")

    counts = df["risk_level"].value_counts().reindex(
        ["Low Risk", "Medium Risk", "High Risk"], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Transactions")
    ax.set_title("Risk Distribution")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)

    st.subheader("إجمالي القيمة حسب نوع العملية")
    by_type = df.groupby("type")["amount"].sum().sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    by_type.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Transaction Type")
    ax2.set_ylabel("Total Amount")
    ax2.set_title("Amount by Transaction Type")
    ax2.tick_params(axis="x", rotation=30)
    st.pyplot(fig2)

    st.subheader("ملخص الأقسام")
    dept = df.groupby("department")["amount"].agg(
        Transactions="count",
        Total="sum",
        Average="mean"
    ).sort_values("Total", ascending=False)
    st.dataframe(dept, use_container_width=True)

with tab2:
    st.subheader("العمليات المحللة")
    st.dataframe(df, use_container_width=True, height=500)

    st.subheader("العمليات عالية المخاطر")
    st.dataframe(
        df[df["risk_level"] == "High Risk"],
        use_container_width=True
    )

with tab3:
    st.subheader("Machine Learning — Anomaly Detection")
    st.write(
        "يستخدم النظام Isolation Forest لاكتشاف العمليات التي تختلف "
        "إحصائيًا عن بقية البيانات. هذه النتيجة مؤشر للمراجعة وليست إثباتًا للاحتيال."
    )

    st.metric("حد العمليات غير المعتادة", f"{unusual_limit:,.2f}")

    ml_df = df[df["ml_anomaly"] == "Yes"]
    st.dataframe(
        ml_df[[
            "transaction_id", "date", "type", "amount",
            "department", "risk_level", "risk_score", "ml_anomaly"
        ]],
        use_container_width=True
    )

with tab4:
    st.subheader("Audit Summary")

    summary = f"""
AI-Audit Audit Summary

Total transactions: {total:,}
Total transaction value: {total_amount:,.2f}
High Risk transactions: {high:,}
Medium Risk transactions: {medium:,}
Low Risk transactions: {int((df["risk_level"] == "Low Risk").sum()):,}
Statistical unusual transactions: {unusual:,}
Machine-learning anomalies: {ml_anomalies:,}

Important note:
The system identifies risk indicators and statistical anomalies.
It does not establish that fraud occurred.
All flagged transactions should be reviewed by a qualified auditor.
"""

    st.text(summary)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Transactions", index=False)
        df.groupby("risk_level").size().rename("count").to_excel(
            writer, sheet_name="Risk Summary"
        )
        df.groupby("type")["amount"].agg(
            Transactions="count",
            Total="sum",
            Average="mean"
        ).to_excel(writer, sheet_name="Type Analysis")
        df.groupby("department")["amount"].agg(
            Transactions="count",
            Total="sum",
            Average="mean"
        ).to_excel(writer, sheet_name="Department Analysis")

    st.download_button(
        "⬇️ تحميل تقرير Excel",
        data=excel_buffer.getvalue(),
        file_name="AI_Audit_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
