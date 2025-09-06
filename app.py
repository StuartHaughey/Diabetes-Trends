# app.py — Diabetes Trends (CareLink CSV/TSV uploads) with simple persistence
# Drop this whole file into your repo and redeploy on Streamlit Cloud.

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

STORE_PATH = "data_store.csv.gz"  # persisted current dataset (parsed & normalised)

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
        # Rebuild datetime/period columns
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df["month"] = df["dt"].dt.to_period("M")
        return df
    except Exception:
        return None

def save_store(df: pd.DataFrame, path: str = STORE_PATH) -> None:
    # Keep only relevant columns to keep the file tidy
    keep_cols = [c for c in df.columns if c in {
        "Date","Time","SG","BG","Bolus","Carbs","Bolus Source","dt","month","source_file"
    }]
    slim = df[keep_cols].copy()
    # month to string for CSV; will be rebuilt on load
    if "month" in slim.columns:
        slim["month"] = slim["month"].astype(str)
    slim.to_csv(path, index=False, compression="gzip")

# ----------------------------
# Robust CareLink file parser
# ----------------------------
@st.cache_data
def parse_file(file) -> pd.DataFrame:
    """
    Reads a CareLink export that may:
      - include a text preamble before the real header
      - use comma, tab, or semicolon delimiters
      - contain BOMs / odd encodings
    Normalises key columns and returns a tidy DataFrame.
    """
    raw_bytes = file.read()
    file.seek(0)

    # Decode with sensible fallbacks
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="ignore")

    # Normalise newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Find the real header line (look for Date + Time, often “Index” appears too)
    header_idx = None
    for i, line in enumerate(lines[:300]):
        s = line.strip("\ufeff ").strip()
        if ("Date" in s and "Time" in s) or re.search(r"\bIndex\b", s):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate a header line with Date/Time in this file.")

    header_line = lines[header_idx]
    # Choose delimiter that yields the most columns on the header
    delim = max([",", "\t", ";"], key=lambda d: len(header_line.split(d)))

    # Read from the header onward
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(body),
        sep=delim,
        engine="python",
        skip_blank_lines=True,
        on_bad_lines="skip",   # tolerate odd rows
    )

    # Drop unnamed filler columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")].copy()

    # Normalise common numeric columns if present
    colmap = {
        "Sensor Glucose (mmol/L)": "SG",
        "BG Reading (mmol/L)": "BG",
        "Bolus Volume Delivered (U)": "Bolus",
        "BWZ Carb Input (grams)": "Carbs",
    }
    for src, dst in colmap.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    # Build datetime columns
    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
            dayfirst=True,   # CareLink often uses dd/mm/yyyy
        )
        df["month"] = df["dt"].dt.to_period("M")

    # Keep a friendly filename for later debugging
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
    use_stored = st.sidebar.toggle("Use stored dataset (ignore new uploads)", value=True)

clr1, clr2 = st.sidebar.columns(2)
with clr1:
    if st.button("Clear stored dataset", help="Deletes the server-side CSV so the app stops auto-loading old data."):
        try:
            if store_exists():
                os.remove(STORE_PATH)
            st.success("Stored dataset cleared. Reload the page.")
        except Exception as e:
            st.error(f"Couldn’t clear store: {e}")

# ----------------------------
# File upload + parsing (if not using stored)
# ----------------------------
data = None

if use_stored and stored is not None:
    data = stored.copy()
else:
    uploaded_files = st.file_uploader(
        "Upload one or more CareLink CSV/TSV exports (monthly or weekly).",
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
        st.info("Upload CSV/TSV files to begin, or save a dataset once to enable stored mode.")
        st.stop()

# ----------------------------
# Deduplicate (safe to merge months over time)
# ----------------------------
# Create a simple signature per row to avoid doubles when you add more months
sig_cols = []
for c in ["dt","SG","BG","Bolus","Carbs","source_file"]:
    if c in data.columns:
        sig_cols.append(c)

if sig_cols:
    data["_sig"] = (
        data[sig_cols]
        .astype(str)
        .agg("|".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
    )
    data = data.drop_duplicates(subset="_sig").drop(columns="_sig")

# ----------------------------
# Guardrails: confirm essentials
# ----------------------------
if "dt" not in data.columns or data["dt"].isna().all():
    st.error("Couldn’t detect Date/Time in the dataset. Ensure the export includes 'Date' and 'Time'.")
    st.stop()

# ----------------------------
# Save control (only shows if we built a dataset from uploads)
# ----------------------------
if not use_stored and data is not None:
    if st.button("💾 Save as current dataset", help="Persist the parsed data so mobile can load it without re-uploading."):
        save_store(data)
        st.success("Saved. Mobile can now load this dataset without uploading.")

st.divider()

# ----------------------------
# Core metrics (overall)
# ----------------------------
sg = pd.to_numeric(data.get("SG"), errors="coerce")
have_sg = sg.notna().sum() > 0

col1, col2, col3, col4 = st.columns(4)

if have_sg:
    mean_sg = sg.mean()
    # GMI (%) for mmol/L:  GMI = 3.31 + 0.43056 * mean_glucose
    gmi = 3.31 + 0.43056 * mean_sg
    tir = ((sg >= 3.9) & (sg <= 10.0)).mean() * 100
    with col1: st.metric("Mean SG (mmol/L)", f"{mean_sg:.2f}")
    with col2: st.metric("GMI (%)", f"{gmi:.2f}")
    with col3: st.metric("Time in Range 3.9–10", f"{tir:.2f}%")
else:
    with col1: st.info("No CGM (SG) values detected.")

# Auto-correction % (MiniMed 780G) using Bolus Source
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
# Monthly trends + tables
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

    # GMI from mean SG
    out["GMI %"] = 3.31 + 0.43056 * out["Mean SG (mmol/L)"]
    out = out.sort_values("month")
    return out

monthly = monthly_summary(data)

st.subheader("Monthly Trends")
if have_sg:
    st.line_chart(monthly.set_index("month")[["Time in Range % (3.9–10)"]])
    st.line_chart(monthly.set_index("month")[["Mean SG (mmol/L)"]])

# Display table with nice month labels & 2-dec rounding
monthly_display = monthly.copy()
monthly_display["month"] = monthly_display["month"].dt.strftime("%b-%Y")
monthly_display = monthly_display.round(2)
st.dataframe(monthly_display, use_container_width=True)

# Download button for computed monthly metrics (rounded)
csv_bytes = monthly_display.to_csv(index=False).encode("utf-8")
st.download_button("Download monthly metrics (CSV)", data=csv_bytes,
                   file_name="diabetes_monthly_metrics.csv", mime="text/csv")

st.divider()

# ----------------------------
# Hour-of-day pattern (quick view)
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
    st.info("Upload files that include CGM (SG) values to see hourly patterns.")

# ----------------------------
# Footer note about persistence
# ----------------------------
st.caption(
    "Tip: Upload on desktop, click “Save as current dataset”, then open this same URL on your phone. "
    "The stored dataset persists across sessions; clearing it or redeploying the app will reset it."
)
