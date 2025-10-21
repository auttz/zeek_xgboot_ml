import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os

# ----------------------------
# 🧭 Page Config
# ----------------------------
st.set_page_config(
    page_title="Zeek XGBoost ML Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Zeek XGBoost ML Dashboard v3")
st.markdown("Monitoring and Insights Dashboard for Zeek ML Pipeline")

# ----------------------------
# 📥 Load Data
# ----------------------------
PREDICT_FILE = "data/output/predict_result.csv"
ARCHIVE_FILE = "data/output/archive_log.txt"

if not os.path.exists(PREDICT_FILE):
    st.warning("⚠️ predict_result.csv not found. Please run the prediction pipeline first.")
    st.stop()

# Load prediction result
df = pd.read_csv(PREDICT_FILE, on_bad_lines='skip')

# Clean column names (if needed)
df.columns = df.columns.str.strip()

# Check for prediction column
if 'prediction' not in df.columns:
    st.error("❌ 'prediction' column not found in predict_result.csv")
    st.stop()

# ----------------------------
# 🧩 Model Summary
# ----------------------------
st.subheader("📊 Model Summary")
total_logs = len(df)
alerts = int(df['prediction'].sum())
normals = total_logs - alerts
alert_ratio = (alerts / total_logs * 100) if total_logs > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Logs", f"{total_logs:,}")
col2.metric("🚨 Alerts", f"{alerts:,} ({alert_ratio:.1f}%)")
col3.metric("✅ Normal", f"{normals:,}")

# ----------------------------
# 📈 Bar Chart (Alerts vs Normal)
# ----------------------------
st.markdown("### 📊 Alert Distribution")

# ใช้ columns ช่วยจัดให้อยู่กึ่งกลางและควบคุมขนาด
col1, col2, col3 = st.columns([1, 2, 1])  # col2 คือพื้นที่หลักของกราฟ
with col2:
    fig, ax = plt.subplots(figsize=(2, 1.2))  # 👈 กำหนดขนาดเล็ก
    ax.bar(["Normal", "Alert"], [normals, alerts],
           color=["#4CAF50", "#FF5252"], width=0.5)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("")  # ไม่ให้ชื่อแกนล่างกินพื้นที่
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, use_container_width=False)

# ----------------------------
# 🕒 Pipeline Run Summary
# ----------------------------
if os.path.exists(ARCHIVE_FILE):
    st.subheader("🕒 Pipeline Run Summary")

    with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    log_df = pd.DataFrame(lines, columns=["raw"])

    # Extract data with regex
    log_df["Timestamp"] = log_df["raw"].str.extract(r"\[(.*?)\]")
    log_df["File"] = log_df["raw"].str.extract(r"Predicted:\s*(.*?),")
    log_df["Accuracy"] = log_df["raw"].str.extract(r"Accuracy:\s*([\d.]+)%").astype(float)
    log_df["Rows"] = log_df["raw"].str.extract(r"Rows:\s*(\d+)").astype(float)
    log_df["Duration"] = log_df["raw"].str.extract(r"Duration:\s*([\d.]+)").astype(float)

    # Show average stats
    if not log_df["Accuracy"].isna().all():
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Avg Accuracy", f"{log_df['Accuracy'].mean():.2f}%")
        c2.metric("🧾 Avg Rows per Run", f"{log_df['Rows'].mean():,.0f}")
        c3.metric("⚡ Avg Duration", f"{log_df['Duration'].mean():.2f} sec")

    # Show last 5 runs
    st.write("📜 **Recent Runs**")
    st.dataframe(log_df[["Timestamp", "File", "Accuracy", "Rows", "Duration"]].tail(5), use_container_width=True)

else:
    st.info("ℹ️ No archive_log.txt found yet — run prediction at least once.")

# ----------------------------
# 🧮 Recent Predictions
# ----------------------------
st.subheader("🧩 Recent Predictions")

# แสดงเฉพาะคอลัมน์สำคัญเพื่อให้อ่านง่าย
cols_to_show = [c for c in df.columns if c in ["@timestamp", "destination.port", "network.protocol", "user_agent.original", "http.request.method", "prediction"]]
if len(cols_to_show) > 0:
    st.dataframe(df[cols_to_show].tail(10), use_container_width=True)
else:
    st.dataframe(df.tail(10), use_container_width=True)

# ----------------------------
# 🕓 Archive Log (Raw)
# ----------------------------
st.subheader("🗂️ Archive Log (Latest 10 Records)")
if os.path.exists(ARCHIVE_FILE):
    archive_lines = open(ARCHIVE_FILE, "r", encoding="utf-8").read().splitlines()
    if archive_lines:
        recent_logs = archive_lines[-10:]
        log_df = pd.DataFrame({
            "Timestamp": [re.search(r"\[(.*?)\]", x).group(1) if re.search(r"\[(.*?)\]", x) else None for x in recent_logs],
            "Event": [x for x in recent_logs]
        })
        st.dataframe(log_df, use_container_width=True)
    else:
        st.write("No logs recorded yet.")
else:
    st.write("No archive_log.txt found.")

# ----------------------------
# 🔄 Refresh Button
# ----------------------------
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ----------------------------
# 🕒 Last Update Time
# ----------------------------
st.caption(f"🕒 Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
