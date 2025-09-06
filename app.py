# app.py — Diabetes Trends (CareLink CSV/TSV uploads)
# New: Analysis window toggle (All data vs Last 12 months, rolling)
# Keeps: 2dp, colour rules, hidden index, Month/Hour first, no inner scroll, persistence.

import io, os, re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

STORE_PATH = "data_store.csv.gz"
DATA_START = pd.Timestamp("2024-01-01")  # ignore anything before this when loading

# ---------- helpers ----------
def df_auto_height(df: pd.DataFrame, row_px: int = 33, header_px: int = 42, max_px: int = 1800) -> int:
    return min(max_px, header_px + row_px * (len(df) + 1))

def store_exists() -> bool:
    return os.path.exists(STORE_PATH) and os.path.getsize(STORE_PATH) > 0

@st.cache_data(show_spinner=False)
def load_store(path: str = STORE_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path, compression="gzip")
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df["month"] = df["dt"].dt.to_period("M")
        return df
    except Exception:
        return None

def save_store(df: pd.DataFrame, path: str = STORE_PATH) -> None:
    keep = [c for c in df.columns if c in {"Date","Time","SG","BG","Bolus","Carbs","Bolus Source","dt","month","source_file"}]
    slim = df[keep].copy()
    if "month" in slim.columns: slim["month"] = slim["month"].astype(str)
    slim.to_csv(path, index=False, compression="gzip")

# ---------- parsing ----------
@st.cache_data
def parse_file(file) -> pd.DataFrame:
    raw = file.read(); file.seek(0)
    try: text = raw.decode("utf-8")
    except UnicodeDecodeError: text = raw.decode("latin-1", errors="ignore")
    text = text.replace("\r\n","\n").replace("\r","\n")
    lines = text.split("\n")

    header_idx = None
    for i, line in enumerate(lines[:300]):
        s = line.strip("\ufeff ").strip()
        if ("Date" in s and "Time" in s) or re.search(r"\bIndex\b", s):
            header_idx = i; break
    if header_idx is None: raise ValueError("Could not locate header line with Date/Time.")

    header_line = lines[header_idx]
    delim = max([",","\t",";"], key=lambda d: len(header_line.split(d)))
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), sep=delim, engine="python", skip_blank_lines=True, on_bad_lines="skip")

    # drop junk cols
    df = df.loc[:, ~df.columns.astype(str).str.match(r"Unnamed")].copy()
    for junk in ("index","level_0","Unnamed: 0"):
        if junk in df.columns: df.drop(columns=[junk], inplace=True)

    # normalise numeric columns if present
    colmap = {
        "Sensor Glucose (mmol/L)": "SG",
        "BG Reading (mmol/L)": "BG",
        "Bolus Volume Delivered (U)": "Bolus",
        "BWZ Carb Input (grams)": "Carbs",
    }
    for src, dst in colmap.items():
        if src in df.columns: df[dst] = pd.to_numeric(df[src], errors="coerce")

    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str), errors="coerce", dayfirst=True)
        df["month"] = df["dt"].dt.to_period("M")

    df["source_file"] = getattr(file, "name", "uploaded_file")
    return df

# ---------- sidebar ----------
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
benchmark_mode = st.sidebar.radio("Colour-coding against:", ["Best practice (guidelines)", "Personal baseline"], index=0)
personal_band = st.sidebar.slider("Personal band (± percentage points)", 2, 10, 5)
show_samples = st.sidebar.toggle("Show hourly 'Samples' column", value=False)

st.sidebar.header("Analysis window")
analysis_mode = st.sidebar.radio(
    "Use data from:",
    ["Last 12 months (rolling)", "All available data"],
    index=0
)

# ---------- load data ----------
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
        st.info("No uploads. Using stored dataset."); data = stored.copy()
    else:
        st.info("Upload CSV/TSV files to begin."); st.stop()

# dedupe + base date fence (>= DATA_START .. today)
if "dt" in data.columns:
    today = pd.Timestamp.today().normalize()
    data = data[(data["dt"] >= DATA_START) & (data["dt"] <= today)]

for junk in ("index","level_0","Unnamed: 0"):
    if junk in data.columns: data.drop(columns=[junk], inplace=True)

sig_cols = [c for c in ("dt","SG","BG","Bolus","Carbs","source_file") if c in data.columns]
if sig_cols:
    data["_sig"] = data[sig_cols].astype(str).agg("|".join, axis=1).str.replace(r"\s+"," ", regex=True)
    data = data.drop_duplicates(subset="_sig").drop(columns="_sig")

if "dt" not in data.columns or data["dt"].isna().all():
    st.error("Couldn’t detect Date/Time in the dataset."); st.stop()

# ---------- apply analysis window ----------
latest_dt = pd.to_datetime(data["dt"]).max()
if analysis_mode.startswith("Last 12"):
    cutoff = (latest_dt - pd.Timedelta(days=365)).normalize()
    analysis = data[data["dt"] >= cutoff].copy()
else:
    cutoff = data["dt"].min().normalize()
    analysis = data.copy()

# Sidebar hint of effective range
earliest_dt = pd.to_datetime(analysis["dt"]).min().date()
latest_dt_display = latest_dt.date()
st.sidebar.caption(f"Analysing: **{earliest_dt} → {latest_dt_display}**")

# Persisted dataset option
if not use_stored and st.button("💾 Save as current dataset"):
    save_store(data); st.success("Saved. Mobile will load this dataset automatically.")

st.divider()

# ---------- metrics ----------
sg = pd.to_numeric(analysis.get("SG"), errors="coerce")
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
if "Bolus" in analysis.columns:
    total_bolus = pd.to_numeric(analysis["Bolus"], errors="coerce").fillna(0)
    src = analysis.get("Bolus Source", pd.Series("", index=analysis.index)).astype(str).str.upper()
    auto_units = total_bolus.where(src.str_contains("AUTO_INSULIN", na=False), 0).sum()
    ac_pct = (auto_units/total_bolus.sum()*100) if total_bolus.sum() else np.nan
    with c4: st.metric("Auto-corrections (% bolus)", f"{ac_pct:.2f}%" if pd.notna(ac_pct) else "—")
else:
    with c4: st.info("No bolus data detected.")

st.divider()

# ---------- monthly summary ----------
analysis["date"]  = analysis["dt"].dt.date
analysis["month"] = analysis["dt"].dt.to_period("M")

def pct_in_range(x, lo, hi):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return ((x>=lo)&(x<=hi)).mean()*100 if len(x) else np.nan

def monthly_summary(df):
    g = df.groupby("month", dropna=True)
    out = pd.DataFrame({
        "Mean SG (mmol/L)": g["SG"].mean(),
        "SD SG (mmol/L)": g["SG"].std(),
        "TIR %": g["SG"].apply(lambda s: pct_in_range(s,3.9,10.0)),
        "TAR % (10–13.9)": g["SG"].apply(lambda s: pct_in_range(s,10.01,13.9)),
        "TAR % (>13.9)": g["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce')>13.9).mean()*100 if s.notna().any() else np.nan),
        "TBR % (3.0–3.9)": g["SG"].apply(lambda s: pct_in_range(s,3.0,3.89)),
        "TBR % (<3.0)": g["SG"].apply(lambda s: (pd.to_numeric(s, errors='coerce')<3.0).mean()*100 if s.notna().any() else np.nan),
        "Bolus Total (U)": g["Bolus"].sum() if "Bolus" in df.columns else np.nan,
        "Carbs Total (g)": g["Carbs"].sum() if "Carbs" in df.columns else np.nan,
    }).reset_index()
    out["GMI %"] = 3.31 + 0.43056*out["Mean SG (mmol/L)"]
    return out.sort_values("month")

monthly = monthly_summary(analysis)

# ---------- colour rules ----------
BEST_PRACTICE = {
    "TIR %": ("gte", 70),
    "TAR % (10–13.9)": ("lte", 25),
    "TAR % (>13.9)": ("lte", 5),
    "TBR % (3.0–3.9)": ("lte", 4),
    "TBR % (<3.0)": ("lte", 1),
    "Mean SG (mmol/L)": ("between", (6.0, 8.0)),
}
def personal_thresholds(df: pd.DataFrame, band_pp: int = 5):
    thr = {}
    if "TIR %" in df.columns:
        base = df["TIR %"].mean(); thr["TIR %"] = ("gte", base + band_pp)
    for col in ["TAR % (10–13.9)","TAR % (>13.9)","TBR % (3.0–3.9)","TBR % (<3.0)"]:
        if col in df.columns:
            base = df[col].mean(); thr[col] = ("lte", max(base - band_pp, 0))
    if "Mean SG (mmol/L)" in df.columns:
        base = df["Mean SG (mmol/L)"].mean(); thr["Mean SG (mmol/L)"] = ("lte", base - 0.2)
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
    return df.style.apply(lambda s: [cell_style(v, s.name) for v in s], axis=0).format(precision=2)

# ---------- charts ----------
st.subheader("Monthly Trends")
if have_sg and len(monthly):
    mplot = monthly.copy()
    mplot["month_str"] = mplot["month"].dt.strftime("%b-%Y")
    mplot = mplot.dropna(subset=["TIR %"])
    if not len(mplot):
        st.info("No valid monthly data to plot.")
    else:
        order = mplot["month_str"].tolist()
        subtitle = "Last 12 months" if analysis_mode.startswith("Last 12") else "All data"
        tir_chart = (
            alt.Chart(mplot).mark_line(point=True)
            .encode(
                x=alt.X("month_str:N", title="Month", sort=order),
                y=alt.Y("TIR %:Q", title="Time in Range %", scale=alt.Scale(domain=[0,100])),
                tooltip=[alt.Tooltip("month_str:N", title="Month"), alt.Tooltip("TIR %:Q", format=".2f")]
            ).properties(height=260, title=f"Time in Range by Month • {subtitle}")
        )
        mean_chart = (
            alt.Chart(mplot).mark_line(point=True)
            .encode(
                x=alt.X("month_str:N", title="Month", sort=order),
                y=alt.Y("Mean SG (mmol/L):Q", title="Mean SG (mmol/L)", scale=alt.Scale(domain=[3,15])),
                tooltip=[alt.Tooltip("month_str:N", title="Month"),
                         alt.Tooltip("Mean SG (mmol/L):Q", format=".2f"),
                         alt.Tooltip("GMI %:Q", title="GMI %", format=".2f")]
            ).properties(height=260, title=f"Mean Glucose by Month • {subtitle}")
        )
        st.altair_chart(tir_chart, use_container_width=True)
        st.altair_chart(mean_chart, use_container_width=True)

# ---------- monthly table ----------
monthly_display = monthly.copy()
monthly_display["month"] = monthly_display["month"].dt.strftime("%b-%Y")
monthly_display = monthly_display.round(2)

tail_cols = [c for c in ["Bolus Total (U)","Carbs Total (g)"] if c in monthly_display.columns]
ordered = ["month","TIR %","TAR % (10–13.9)","TAR % (>13.9)","TBR % (3.0–3.9)","TBR % (<3.0)",
           "Mean SG (mmol/L)","SD SG (mmol/L)"] + tail_cols
ordered = [c for c in ordered if c in monthly_display.columns]
monthly_display = monthly_display[ordered]

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

# ---------- hourly pattern ----------
st.subheader("Hour-of-day pattern (combined)")
if have_sg:
    tmp = analysis[["dt","SG"]].dropna().copy()
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

    cols = ["hour","Time in Range %","Hyper % (>10)","Severe Hyper % (>13.9)","Hypo % (<3.9)"] + (["Samples"] if show_samples else [])
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
    "Tables show 2 decimal places. Analysis window is set in the sidebar (Last 12 months or All data). "
    "Benchmark colours: Green meets target • Amber near target • Red outside target."
)
