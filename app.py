import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Diabetes Trends", layout="wide")

st.title("📊 Diabetes Trends (CareLink CSV uploads)")

uploaded_files = st.file_uploader(
    "Upload one or more CareLink CSV/TSV exports",
    type=["csv", "tsv"],
    accept_multiple_files=True
)

@st.cache_data
def parse_file(file) -> pd.DataFrame:
    # Detect delimiter
    sample = file.read().decode("utf-8", errors="ignore")
    file.seek(0)
    if "\t" in sample.splitlines()[0]:
        delim = "\t"
    elif ";" in sample.splitlines()[0]:
        delim = ";"
    else:
        delim = ","
    df = pd.read_csv(file, sep=delim, engine="python")
    # Standardise cols
    if "Sensor Glucose (mmol/L)" in df.columns:
        df["SG"] = pd.to_numeric(df["Sensor Glucose (mmol/L)"], errors="coerce")
    if "BG Reading (mmol/L)" in df.columns:
        df["BG"] = pd.to_numeric(df["BG Reading (mmol/L)"], errors="coerce")
    if "Bolus Volume Delivered (U)" in df.columns:
        df["Bolus"] = pd.to_numeric(df["Bolus Volume Delivered (U)"], errors="coerce")
    if "BWZ Carb Input (grams)" in df.columns:
        df["Carbs"] = pd.to_numeric(df["BWZ Carb Input (grams)"], errors="coerce")
    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce", dayfirst=True)
        df["month"] = df["dt"].dt.to_period("M")
    return df

if uploaded_files:
    frames = []
    for file in uploaded_files:
        frames.append(parse_file(file))
    data = pd.concat(frames, ignore_index=True)

    st.success(f"Loaded {len(uploaded_files)} files, {len(data)} rows.")

    # --- Metrics ---
    sg = data["SG"].dropna()
    if len(sg):
        mean_sg = sg.mean()
        gmi = 3.31 + 0.43056 * mean_sg
        tir = ((sg >= 3.9) & (sg <= 10)).mean() * 100
        st.metric("Mean SG (mmol/L)", f"{mean_sg:.1f}")
        st.metric("GMI %", f"{gmi:.2f}")
        st.metric("Time in Range (3.9–10)", f"{tir:.1f}%")

    # --- Monthly trends ---
    monthly = data.groupby("month").agg(
        mean_SG=("SG","mean"),
        TIR=("SG", lambda x: ((x>=3.9)&(x<=10)).mean()*100),
        carbs_day=("Carbs", lambda x: x.sum()/data["dt"].dt.date.nunique() if len(x) else 0),
        bolus_total=("Bolus","sum"),
    )
    st.subheader("Monthly Trends")
    st.line_chart(monthly[["TIR"]])
    st.line_chart(monthly[["mean_SG"]])

    st.dataframe(monthly)

else:
    st.info("Upload CSV/TSV files to begin.")
