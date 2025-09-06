# app.py — Diabetes Trends (CareLink CSV/TSV uploads)
# Drop this whole file into your repo and redeploy.

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

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
# File upload + parsing
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more CareLink CSV/TSV exports (monthly or weekly).",
    type=["csv", "tsv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV/TSV files to begin.")
    st.stop()

frames, bad = [], []
for f in uploaded_files:
    try:
        frames.append(parse_file(f))
    except Exception:
        bad.append(f.name)

if bad:
    st.warning("Skipped files that failed to parse: " + ", ".join(bad))

if not frames:
    st.error("No valid files parsed. Please check your exports.")
    st.stop()

data = pd.concat(frames, ignore_index=True)
st.success(f"Loaded {len(frames)} file(s) • {len(data):,} rows")

# ----------------------------
# Guardrails: confirm essentials
# ----------------------------
if "dt" not in data.columns or data["dt"].isna().all():
    st.error("Couldn’t detect Date/Time in the uploads. Please ensure the export includes 'Date' and 'Time' columns.")
    st.stop()

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

# Auto-correction % (MiniMed 780G) using Bolus Source and carb entries
if "Bolus" in data.columns:
    total_bolus = pd.to_numeric(data["Bolus"], errors="coerce").fillna(0)
    carbs_col = pd.to_numeric(data.get("Carbs"), errors="coerce").fillna(0) if "Carbs" in data.columns else pd.Series(0, index=data.index, dtype=float)
    # Identify auto-corrections via Bolus Source flag
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
data["date"] = data["dt"].dt.date
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
    out = out.rename(columns={
    "mean_SG_mmol/L": "Mean SG (mmol/L)",
    "sd_SG_mmol/L": "SD SG (mmol/L)",
    "TIR_% (3.9–10)": "Time in Range % (3.9–10)",
    "TAR_% (10–13.9)": "Time Above Range % (10–13.9)",
    "TAR_% (>13.9)": "Time Above Range % (>13.9)",
    "TBR_% (3.0–3.9)": "Time Below Range % (3.0–3.9)",
    "TBR_% (<3.0)": "Time Below Range % (<3.0)",
    "Bolus_total_U": "Bolus Total (U)",
    "Carbs_total_g": "Carbs Total (g)",
    "GMI_%": "GMI %"
})

    return out


monthly = monthly_summary(data)
monthly_display = monthly.copy()
# Reformat Period to nice string like Jan-2025
monthly_display["month"] = monthly_display["month"].dt.strftime("%b-%Y")
monthly_display = monthly_display.round(2)


st.subheader("Monthly Trends")
if have_sg:
    # Charts expect numeric index; keep month order by using the original Period index
    st.line_chart(monthly.set_index("month")[["TIR_% (3.9–10)"]])
    st.line_chart(monthly.set_index("month")[["mean_SG_mmol/L"]])
st.dataframe(monthly_display, use_container_width=True)

# Download button for computed monthly metrics (rounded to 2 decimals)
csv_bytes = monthly_display.to_csv(index=False).encode("utf-8")
st.download_button("Download monthly metrics (CSV)", data=csv_bytes, file_name="diabetes_monthly_metrics.csv", mime="text/csv")

st.divider()

# ----------------------------
# Hour-of-day pattern (quick view)
# ----------------------------
st.subheader("Hour-of-day pattern (combined uploads)")
if have_sg:
    tmp = data[["dt", "SG"]].dropna().copy()
    tmp["hour"] = tmp["dt"].dt.hour
    hourly = tmp.groupby("hour").agg(
    **{
        "Time in Range %": ("SG", lambda s: ((s>=3.9)&(s<=10.0)).mean()*100),
        "Hyper % (>10)": ("SG", lambda s: (s>10.0).mean()*100),
        "Severe Hyper % (>13.9)": ("SG", lambda s: (s>13.9).mean()*100),
        "Hypo % (<3.9)": ("SG", lambda s: (s<3.9).mean()*100),
        "Samples": ("SG","count")
    }
).reset_index()
hourly = hourly.rename(columns={
    "TIR_pct": "Time in Range %",
    "Hyper_pct": "Hyper % (>10)",
    "SevereHyper_pct": "Severe Hyper % (>13.9)",
    "Hypo_pct": "Hypo % (<3.9)",
    "samples": "Samples"
})

    hourly = hourly.round(2)
    st.dataframe(hourly, use_container_width=True)
else:
    st.info("Upload files that include CGM (SG) values to see hourly patterns.")
