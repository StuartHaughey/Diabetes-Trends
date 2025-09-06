# app.py — Diabetes Trends (CareLink CSV/TSV uploads) with future-date filtering

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

STORE_PATH = "data_store.csv.gz"

# ----------------------------
# Helpers: load/save store
# ----------------------------
def store_exists() -> bool:
    return os.path.exists(STORE_PATH) and os.path.getsize(STORE_PATH) > 0

@st.cache_data(show_spinner=False)
def load_store(path: str = STORE_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, compression="gzip")
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df["month"] = df["dt"].dt.to_period("M")
        return df
    except Exception:
        return None

def save_store(df: pd.DataFrame, path: str = STORE_PATH) -> None:
    keep_cols = [c for c in df.columns if c in {
        "Date","Time","SG","BG","Bolus","Carbs","Bolus Source","dt","month","source_file"
    }]
    slim = df[keep_cols].copy()
    if "month" in slim.columns:
        slim["month"] = slim["month"].astype(str)
    slim.to_csv(path, index=False, compression="gzip")

# ----------------------------
# Robust CareLink file parser
# ----------------------------
@st.cache_data
def parse_file(file) -> pd.DataFrame:
    raw_bytes = file.read()
    file.seek(0)

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="ignore")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    header_idx = None
    for i, line in enumerate(lines[:300]):
        s = line.strip("\ufeff ").strip()
        if ("Date" in s and "Time" in s) or re.search(r"\bIndex\b", s):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate a header line with Date/Time in this file.")

    header_line = lines[header_idx]
    delim = max([",", "\t", ";"], key=lambda d: len(header_line.split(d)))

    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), sep=delim, engine="python",
                     skip_blank_lines=True, on_bad_lines="skip")

    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")].copy()

    colmap = {
        "Sensor Glucose (mmol/L)": "SG",
        "BG Reading (mmol/L)": "BG",
        "Bolus Volume Delivered (U)": "Bolus",
        "BWZ Carb Input (grams)": "Carbs",
    }
    for src, dst in colmap.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                  errors="coerce", dayfirst=True)
        df["month"] = df["dt"].dt.to_period("M")

    df["source_file"] = getattr(file, "name", "uploaded_file")
    return df

# ----------------------------
# Sidebar: choose data source
# ----------------------------
st.sidebar.header("Data source")
stored = load_store() if store_exists() else None
use_stored = False

if stored is not None:
    st.sidebar.success(f"Stored dataset found: {len(stored):,} rows")
    use_stored = st.sidebar.toggle("Use stored dataset", value=True)

if st.sidebar.button("Clear stored dataset"):
    try:
        if store_exists():
            os.remove(STORE_PATH)
        st.success("Stored dataset cleared. Reload the page.")
    except Exception as e:
        st.error(f"Couldn’t clear store: {e}")

# ----------------------------
# Load data
# ----------------------------
data = None

if use_stored and stored is not None:
    data = stored.copy()
else:
    uploaded_files = st.file_uploader(
        "Upload one or more CareLink CSV/TSV exports",
        type=["csv", "tsv"],
        accept_multiple_files=True
    )

    frames, bad = [], []
    if uploaded_files:
        for f in uploaded_files:
            try:
                frames.append(parse_file(f))
            except Exception:
                bad.append(f.name)

    if bad:
        st.warning("Skipped files that failed to parse: " + ", ".join(bad))

    if frames:
        data = pd.concat(frames, ignore_index=True)
        st.success(f"Loaded {len(frames)} file(s) • {len(data):,} rows")
    elif stored is not None:
        st.info("No files uploaded. Falling back to stored dataset.")
        data = stored.copy()
    else:
        st.info("Upload CSV/TSV files to begin.")
        st.stop()

# ----------------------------
# Deduplicate and filter future dates
# ----------------------------
if "dt" in data.columns:
    today = pd.Timestamp.today().normalize()
    data = data[data["dt"] <= today]

sig_cols = [c for c in ["dt","SG","BG","Bolus","Carbs","source_file"] if c in data.columns]
if sig_cols:
    data["_sig"] = data[sig_cols].astype(str).agg("|".join, axis=1).str.replace(r"\s+", " ", regex=True)
    data = data.drop_duplicates(subset="_sig").drop(columns="_sig")

if "dt" not in data.columns or data["dt"].isna().all():
    st.error("Couldn’t detect Date/Time in the dataset.")
    st.stop()

if not use_stored and data is not None:
    if st.button("💾 Save as current dataset"):
        save_store(data)
        st.success("Saved. Mobile can now load this dataset without uploading.")

st.divider()

# ----------------------------
# Core metrics
# ----------------------------
sg = pd.to_numeric(data.get("SG"), errors="coerce")
have_sg = sg.notna().sum() > 0

col1, col2, col3, col4 = st.columns(4)

if have_sg:
    mean_sg = sg.mean()
    gmi = 3.31 + 0.43056 * mean_sg
    tir = ((sg >= 3.9) & (sg <= 10.0)).mean() * 100
    with col1: st.metric("Mean SG (mmol/L)", f"{mean_sg:.2f}")
    with col2: st.metric("GMI (%)", f"{gmi:.2f}")
    with col3: st.metric("Time in Range 3.9–10", f"{tir:.2f}%")
else:
    with col1: st.info("No CGM values detected.")

if "Bolus" in data.columns:
    total_bolus = pd.to_numeric(data["Bolus"], errors="coerce").fillna(0)
    src = data.get("Bolus Source", pd.Series("", index=data.index)).astype(str).str.upper()
    auto_units = total_bolus.where(src.str.contains("AUTO_INSULIN"), 0).sum()
    autocorr_pct = (auto_units / total_bolus.sum() * 100) if total_bolus.sum() else np.nan
    with col4: st.metric("Auto-corrections (% bolus)", f"{autocorr_pct:.2f}%" if pd.notna(autocorr_pct) else "—")
else:
    with col4: st.info("No bolus data detected.")

st.divider()

# ----------------------------
# Monthly trends
# ----------------------------
data["date"]  = data["dt"].dt.date
data["month"] = data["dt"].dt.to_period("M")

def pct_in_range(x, lo, hi):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return ((x >= lo) & (x <= hi)).mean() * 100 if len(x) else np.nan

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("month", dropna=True)
    out = pd.DataFrame({
        "Mean SG (mmol/L)": grp["SG"].mean(),
        "SD SG (mmol/L)": grp["SG"].std(),
        "Time in Range % (3.9–10)": grp["SG"].apply(lambda s: pct_in_range(s, 3.9, 10.0)),
        "Time Above Range % (10–13.9)": grp["SG"].apply(lambda s: pct_in_range(s, 10.01, 13.9)),
        "Time Above Range % (>13.9)": grp["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce') > 13.9).mean()*100 if s.notna().any() else np.nan),
        "Time Below Range % (3.0–3.9)": grp["SG"].apply(lambda s: pct_in_range(s, 3.0, 3.89)),
        "Time Below Range % (<3.0)": grp["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce') < 3.0).mean()*100 if s.notna().any() else np.nan),
        "Bolus Total (U)": grp["Bolus"].sum() if "Bolus" in df.columns else np.nan,
        "Carbs Total (g)": grp["Carbs"].sum() if "Carbs" in df.columns else np.nan,
    }).reset_index()

    out["GMI %"] = 3.31 + 0.43056 * out["Mean SG (mmol/L)"]
    return out.sort_values("month")

monthly = monthly_summary(data)

st.subheader("Monthly Trends")
if have_sg:
    st.subheader("Monthly Trends")
if have_sg and len(monthly):
    mplot = monthly.copy()
    mplot["month_str"] = mplot["month"].dt.strftime("%b-%Y")
    # keep original order
    month_order = mplot["month_str"].tolist()

    tir_chart = (
        alt.Chart(mplot)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_str:N", title="Month", sort=month_order),
            y=alt.Y("Time in Range % (3.9–10):Q",
                    title="Time in Range %", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("month_str:N", title="Month"),
                alt.Tooltip("Time in Range % (3.9–10):Q", format=".2f"),
            ],
        )
        .properties(height=260, title="Time in Range by Month")
    )

    mean_chart = (
        alt.Chart(mplot)
        .mark_line(point=True)
        .encode(
            x=alt.X("month_str:N", title="Month", sort=month_order),
            y=alt.Y("Mean SG (mmol/L):Q",
                    title="Mean Sensor Glucose (mmol/L)",
                    scale=alt.Scale(domain=[3, 15])),
            tooltip=[
                alt.Tooltip("month_str:N", title="Month"),
                alt.Tooltip("Mean SG (mmol/L):Q", format=".2f"),
                alt.Tooltip("GMI %:Q", title="GMI %", format=".2f"),
            ],
        )
        .properties(height=260, title="Mean Glucose by Month")
    )

    st.altair_chart(tir_chart, use_container_width=True)
    st.altair_chart(mean_chart, use_container_width=True)
else:
    st.info("Not enough monthly data to plot yet.")


monthly_display = monthly.copy()
monthly_display["month"] = monthly_display["month"].dt.strftime("%b-%Y")
monthly_display = monthly_display.round(2)
st.dataframe(monthly_display, use_container_width=True)

csv_bytes = monthly_display.to_csv(index=False).encode("utf-8")
st.download_button("Download monthly metrics (CSV)", data=csv_bytes,
                   file_name="diabetes_monthly_metrics.csv", mime="text/csv")

st.divider()

# ----------------------------
# Hour-of-day pattern
# ----------------------------
st.subheader("Hour-of-day pattern (combined)")
if have_sg:
    tmp = data[["dt", "SG"]].dropna().copy()
    tmp["hour"] = tmp["dt"].dt.hour
    hourly = tmp.groupby("hour").agg(
        **{
            "Time in Range %": ("SG", lambda s: ((s>=3.9)&(s<=10.0)).mean()*100),
            "Hyper % (>10)": ("SG", lambda s: (s>10.0).mean()*100),
            "Severe Hyper % (>13.9)": ("SG", lambda s: (s>13.9).mean()*100),
            "Hypo % (<3.9)": ("SG", lambda s: (s<3.9).mean()*100),
            "Samples": ("SG", "count"),
        }
    ).reset_index()
    hourly = hourly.round(2)
    st.dataframe(hourly, use_container_width=True)
else:
    st.info("Upload files with CGM values to see hourly patterns.")

st.caption("Rows with future dates are automatically excluded.")
