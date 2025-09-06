# app.py — Diabetes Trends (CareLink CSV/TSV uploads)
# Fixes:
# - Monthly TIR plot uses DATA_START..today (no hard-coded 2025+)
# - Hide index cleanly; Month/Hour first; optional Samples toggle
# - Drop stray columns (index/level_0/Unnamed)
# - 2dp formatting, colour rules, persistence preserved

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

STORE_PATH = "data_store.csv.gz"
DATA_START = pd.Timestamp("2024-01-01")  # fence the analysis window

# ----------------------------
# Small UI helpers
# ----------------------------
def df_auto_height(df: pd.DataFrame, row_px: int = 33, header_px: int = 42, max_px: int = 1800) -> int:
    return min(max_px, header_px + row_px * (len(df) + 1))

# ----------------------------
# Persistence helpers
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
# Robust CareLink parser
# ----------------------------
@st.cache_data
def parse_file(file) -> pd.DataFrame:
    raw = file.read(); file.seek(0)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")
    text = text.replace("\r\n","\n").replace("\r","\n")
    lines = text.split("\n")

    header_idx = None
    for i, line in enumerate(lines[:300]):
        s = line.strip("\ufeff ").strip()
        if ("Date" in s and "Time" in s) or re.search(r"\bIndex\b", s):
            header_idx = i; break
    if header_idx is None:
        raise ValueError("Could not locate header line with Date/Time.")

    header_line = lines[header_idx]
    delim = max([",","\t",";"], key=lambda d: len(header_line.split(d)))
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), sep=delim, engine="python",
                     skip_blank_lines=True, on_bad_lines="skip")

    # Drop junk columns up front
    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")].copy()
    for junk in ["index", "level_0", "Unnamed: 0"]:
        if junk in df.columns: df = df.drop(columns=[junk])

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
        df["dt"] = pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str),
                                  errors="coerce", dayfirst=True)
        df["month"] = df["dt"].dt.to_period("M")

    df["source_file"] = getattr(file, "name", "uploaded_file")
    return df

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Data")
stored = load_store() if store_exists() else None
use_stored = False
if stored is not None:
    st.sidebar.success(f"Stored dataset: {len(stored):,} rows")
    use_stored = st.sidebar.toggle("Use stored dataset", value=True)
if st.sidebar.button("Clear stored dataset"):
    try:
        if store_exists(): os.remove(STORE_PATH)
        st.success("Stored dataset cleared. Reload the page.")
    except Exception as e:
        st.error(f"Couldn’t clear store: {e}")

st.sidebar.header("Benchmark")
benchmark_mode = st.sidebar.radio(
    "Colour-coding against:",
    ["Best practice (guidelines)", "Personal baseline"],
    index=0
)
personal_band = st.sidebar.slider("Personal band (± percentage points)", 2, 10, 5)

show_samples = st.sidebar.toggle("Show hourly 'Samples' column", value=False)

# ----------------------------
# Load data (uploads or stored)
# ----------------------------
if use_stored and stored is not None:
    data = stored.copy()
else:
    files = st.file_uploader("Upload CareLink CSV/TSV exports", type=["csv","tsv"], accept_multiple_files=True)
    frames, bad = [], []
    if files:
        for f in files:
            try: frames.append(parse_file(f))
            except Exception: bad.append(f.name)
    if bad: st.warning("Skipped files: " + ", ".join(bad))
    if frames:
        data = pd.concat(frames, ignore_index=True)
        st.success(f"Loaded {len(frames)} file(s) • {len(data):,} rows")
    elif stored is not None:
        st.info("No uploads. Using stored dataset.")
        data = stored.copy()
    else:
        st.info("Upload CSV/TSV files to begin.")
        st.stop()

# ----------------------------
# Deduplicate + date fence
# ----------------------------
if "dt" in data.columns:
    today = pd.Timestamp.today().normalize()
    data = data[(data["dt"] >= DATA_START) & (data["dt"] <= today)]

# drop residual junk columns if any got in via store
for junk in ["index", "level_0", "Unnamed: 0"]:
    if junk in data.columns: data = data.drop(columns=[junk])

sig_cols = [c for c in ["dt","SG","BG","Bolus","Carbs","source_file"] if c in data.columns]
if sig_cols:
    data["_sig"] = data[sig_cols].astype(str).agg("|".join, axis=1).str.replace(r"\s+"," ", regex=True)
    data = data.drop_duplicates(subset="_sig").drop(columns="_sig")

if "dt" not in data.columns or data["dt"].isna().all():
    st.error("Couldn’t detect Date/Time in the dataset."); st.stop()

if not use_stored and st.button("💾 Save as current dataset"):
    save_store(data); st.success("Saved. Mobile will load this dataset automatically.")

st.divider()

# ----------------------------
# Metrics
# ----------------------------
sg = pd.to_numeric(data.get("SG"), errors="coerce")
have_sg = sg.notna().sum() > 0
c1,c2,c3,c4 = st.columns(4)
if have_sg:
    mean_sg = sg.mean()
    gmi = 3.31 + 0.43056*mean_sg
    tir = ((sg>=3.9)&(sg<=10)).mean()*100
    with c1: st.metric("Mean SG (mmol/L)", f"{mean_sg:.2f}")
    with c2: st.metric("GMI (%)", f"{gmi:.2f}")
    with c3: st.metric("Time in Range 3.9–10", f"{tir:.2f}%")
else:
    with c1: st.info("No CGM values detected.")
if "Bolus" in data.columns:
    total_bolus = pd.to_numeric(data["Bolus"], errors="coerce").fillna(0)
    src = data.get("Bolus Source", pd.Series("", index=data.index)).astype(str).str.upper()
    auto_units = total_bolus.where(src.str.contains("AUTO_INSULIN"), 0).sum()
    ac_pct = (auto_units/total_bolus.sum()*100) if total_bolus.sum() else np.nan
    with c4: st.metric("Auto-corrections (% bolus)", f"{ac_pct:.2f}%" if pd.notna(ac_pct) else "—")
else:
    with c4: st.info("No bolus data detected.")

st.divider()

# ----------------------------
# Monthly summary
# ----------------------------
data["date"]  = data["dt"].dt.date
data["month"] = data["dt"].dt.to_period("M")

def pct_in_range(x, lo, hi):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return ((x>=lo)&(x<=hi)).mean()*100 if len(x) else np.nan

def monthly_summary(df):
    g = df.groupby("month", dropna=True)
    out = pd.DataFrame({
        "Mean SG (mmol/L)": g["SG"].mean(),
        "SD SG (mmol/L)": g["SG"].std(),
        "Time in Range % (3.9–10)": g["SG"].apply(lambda s: pct_in_range(s,3.9,10)),
        "Time Above Range % (10–13.9)": g["SG"].apply(lambda s: pct_in_range(s,10.01,13.9)),
        "Time Above Range % (>13.9)": g["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce')>13.9).mean()*100 if s.notna().any() else np.nan),
        "Time Below Range % (3.0–3.9)": g["SG"].apply(lambda s: pct_in_range(s,3.0,3.89)),
        "Time Below Range % (<3.0)": g["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce')<3.0).mean()*100 if s.notna().any() else np.nan),
        "Bolus Total (U)": g["Bolus"].sum() if "Bolus" in df.columns else np.nan,
        "Carbs Total (g)": g["Carbs"].sum() if "Carbs" in df.columns else np.nan,
    }).reset_index()
    out["GMI %"] = 3.31 + 0.43056*out["Mean SG (mmol/L)"]
    return out.sort_values("month")

monthly = monthly_summary(data)

# ----------------------------
# Colour rules
# ----------------------------
BEST_PRACTICE = {
    "Time in Range % (3.9–10)": ("gte", 70),
    "Time Above Range % (10–13.9)": ("lte", 25),
    "Time Above Range % (>13.9)": ("lte", 5),
    "Time Below Range % (3.0–3.9)": ("lte", 4),
    "Time Below Range % (<3.0)": ("lte", 1),
    "Mean SG (mmol/L)": ("between", (6.0, 8.0)),
}

def personal_thresholds(df: pd.DataFrame, band_pp: int = 5):
    thr = {}
    if "Time in Range % (3.9–10)" in df.columns:
        base = df["Time in Range % (3.9–10)"].mean()
        thr["Time in Range % (3.9–10)"] = ("gte", base + band_pp)
    for col in ["Time Above Range % (10–13.9)","Time Above Range % (>13.9)",
                "Time Below Range % (3.0–3.9)","Time Below Range % (<3.0)"]:
        if col in df.columns:
            base = df[col].mean()
            thr[col] = ("lte", max(base - band_pp, 0))
    if "Mean SG (mmol/L)" in df.columns:
        base = df["Mean SG (mmol/L)"].mean()
        thr["Mean SG (mmol/L)"] = ("lte", base - 0.2)
    return thr

GREEN = "background-color:#c6efce;color:#006100"
RED   = "background-color:#ffc7ce;color:#9c0006"
AMBER = "background-color:#fff2cc;color:#7f6000"
CLEAR = ""

def style_by_rules(df: pd.DataFrame, mode: str):
    rules = BEST_PRACTICE if mode.startswith("Best") else personal_thresholds(df, personal_band)

    def cell_style(val, col):
        if pd.isna(val): return CLEAR
        rule = rules.get(col)
        if not rule: return CLEAR
        kind, target = rule
        if kind == "gte":
            if val >= target: return GREEN
            if val >= max(target-5, 0): return AMBER
            return RED
        if kind == "lte":
            if val <= target: return GREEN
            if val <= target + 5: return AMBER
            return RED
        if kind == "between":
            lo, hi = target
            if lo <= val <= hi: return GREEN
            if (lo-0.5) <= val <= (hi+0.5): return AMBER
            return RED
        return CLEAR

    # return a Styler with colours + 2dp; we'll also hide index via st.dataframe(hide_index=True)
    return df.style.apply(lambda s: [cell_style(v, s.name) for v in s], axis=0).format(precision=2)

# ----------------------------
# Monthly charts (Altair)
# ----------------------------
st.subheader("Monthly Trends")
if have_sg and len(monthly):
    mplot = monthly.copy()
    mplot["month_str"] = mplot["month"].dt.strftime("%b-%Y")
    # Plot months within the same fence as the data
    mplot = mplot[(mplot["month"].dt.to_timestamp() >= DATA_START)]
    mplot = mplot.dropna(subset=["Time in Range % (3.9–10)"])
    if not len(mplot):
        st.info("No valid monthly data to plot.")
    else:
        order = mplot["month_str"].tolist()
        tir_chart = (
            alt.Chart(mplot).mark_line(point=True)
            .encode(
                x=alt.X("month_str:N", title="Month", sort=order),
                y=alt.Y("Time in Range % (3.9–10):Q", title="Time in Range %",
                        scale=alt.Scale(domain=[0,100])),
                tooltip=[alt.Tooltip("month_str:N", title="Month"),
                         alt.Tooltip("Time in Range % (3.9–10):Q", format=".2f")]
            ).properties(height=260, title="Time in Range by Month")
        )
        mean_chart = (
            alt.Chart(mplot).mark_line(point=True)
            .encode(
                x=alt.X("month_str:N", title="Month", sort=order),
                y=alt.Y("Mean SG (mmol/L):Q", title="Mean SG (mmol/L)",
                        scale=alt.Scale(domain=[3,15])),
                tooltip=[alt.Tooltip("month_str:N", title="Month"),
                         alt.Tooltip("Mean SG (mmol/L):Q", format=".2f"),
                         alt.Tooltip("GMI %:Q", title="GMI %", format=".2f")]
            ).properties(height=260, title="Mean Glucose by Month")
        )
        st.altair_chart(tir_chart, use_container_width=True)
        st.altair_chart(mean_chart, use_container_width=True)

# ----------------------------
# Monthly table (2dp, coloured, Month first, no index)
# ----------------------------
monthly_display = monthly.copy()
monthly_display["month"] = monthly_display["month"].dt.strftime("%b-%Y")
monthly_display = monthly_display.round(2)

# Ensure Month is first, and push totals to the end so “counts” aren’t first
tail_cols = [c for c in ["Bolus Total (U)","Carbs Total (g)"] if c in monthly_display.columns]
main_cols = [c for c in monthly_display.columns if c not in (["month"] + tail_cols)]
cols = ["month"] + main_cols[1:] + tail_cols  # keep month, then metrics, then totals
monthly_display = monthly_display[cols]

st.dataframe(
    style_by_rules(monthly_display, benchmark_mode),
    use_container_width=True,
    height=df_auto_height(monthly_display),
    hide_index=True
)

csv_bytes = monthly_display.round(2).to_csv(index=False).encode("utf-8")
st.download_button("Download monthly metrics (CSV)", data=csv_bytes,
                   file_name="diabetes_monthly_metrics.csv", mime="text/csv")

st.divider()

# ----------------------------
# Hour-of-day pattern (2dp, colours; Hour first; optional Samples; no index)
# ----------------------------
st.subheader("Hour-of-day pattern (combined)")
if have_sg:
    tmp = data[["dt","SG"]].dropna().copy()
    tmp["hour"] = tmp["dt"].dt.hour
    hourly = tmp.groupby("hour").agg(
        **{
            "Time in Range %": ("SG", lambda s: ((s>=3.9)&(s<=10.0)).mean()*100),
            "Hyper % (>10)": ("SG", lambda s: (s>10.0).mean()*100),
            "Severe Hyper % (>13.9)": ("SG", lambda s: (s>13.9).mean()*100),
            "Hypo % (<3.9)": ("SG", lambda s: (s<3.9).mean()*100),
            "Samples": ("SG","count"),
        }
    ).reset_index()

    hourly = hourly.round(2)

    GREEN = "background-color:#c6efce;color:#006100"
    RED   = "background-color:#ffc7ce;color:#9c0006"
    AMBER = "background-color:#fff2cc;color:#7f6000"

    def style_hourly(df):
        def color(val, col):
            if pd.isna(val): return ""
            if "Time in Range" in col:
                if val >= 70: return GREEN
                if val >= 50: return AMBER
                return RED
            if "Hypo" in col:
                if val <= 4: return GREEN
                if val <= 6: return AMBER
                return RED
            if "Hyper" in col:
                thr = 5 if ">13.9" in col else 25
                if val <= thr: return GREEN
                if val <= thr + 5: return AMBER
                return RED
            return ""
        return df.style.apply(lambda s: [color(v, s.name) for v in s], axis=0).format(precision=2)

    # Build column order (Hour first, Samples last or hidden)
    metric_cols = ["Time in Range %","Hyper % (>10)","Severe Hyper % (>13.9)","Hypo % (<3.9)"]
    cols = ["hour"] + metric_cols + (["Samples"] if show_samples else [])
    hourly = hourly[cols]

    st.dataframe(
        style_hourly(hourly),
        use_container_width=True,
        height=df_auto_height(hourly),
        hide_index=True
    )
else:
    st.info("Upload files with CGM values to see hourly patterns.")

st.caption(
    "Tables show 2 decimal places. Data limited to dates from 01-Jan-2024 through today. "
    "Benchmark colours: Green meets target • Amber near target • Red outside target."
)
