# app.py — Diabetes Trends with One-Page PDF Export
# New: Export to PDF (one page) with toggle to include short commentary.
# Keeps: 3/6/12/all windows, 2dp formatting, colour rules, Month/Hour first, no inner scroll, persistence.

import io, os, re, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# For PDF & charts used in PDF (graceful fallback if not installed)
PDF_AVAILABLE = True
PDF_IMPORT_ERROR = ""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
except Exception as _e:
    PDF_AVAILABLE = False
    PDF_IMPORT_ERROR = str(_e)


st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("📊 Diabetes Trends (CareLink CSV/TSV uploads)")

STORE_PATH = "data_store.csv.gz"
DATA_START = pd.Timestamp("2024-01-01")  # ignore anything before this when loading

# ---------- small helpers ----------
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
    ["Last 3 months", "Last 6 months", "Last 12 months (rolling)", "All available data"],
    index=2
)

st.sidebar.header("Export")
include_comments = st.sidebar.checkbox("Include commentary in export", value=True)
patient_name = st.sidebar.text_input("Patient name (for PDF header)", value="Stuart Haughey")

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

def window_start(mode: str, latest: pd.Timestamp) -> pd.Timestamp:
    if mode.startswith("Last 3"):  return (latest - pd.Timedelta(days=90)).normalize()
    if mode.startswith("Last 6"):  return (latest - pd.Timedelta(days=182)).normalize()
    if mode.startswith("Last 12"): return (latest - pd.Timedelta(days=365)).normalize()
    return data["dt"].min().normalize()

cutoff = window_start(analysis_mode, latest_dt)
analysis = data.copy() if analysis_mode.startswith("All") else data[data["dt"] >= cutoff].copy()

window_label = (
    "All data"
    if analysis_mode.startswith("All")
    else f"{analysis_mode} (from {cutoff.date()} to {latest_dt.date()})"
)

# Sidebar range hint
earliest_dt = pd.to_datetime(analysis["dt"]).min().date()
st.sidebar.caption(f"Analysing: **{earliest_dt} → {latest_dt.date()}**")

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
    # robust Bolus Source handling
    if "Bolus Source" in analysis.columns:
        src = analysis["Bolus Source"].astype(str).str.upper()
    else:
        src = pd.Series("", index=analysis.index)
    auto_units = total_bolus.where(src.str.contains("AUTO_INSULIN", na=False), 0).sum()
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

# ---------- charts on page ----------
st.subheader("Monthly Trends")
if have_sg and len(monthly):
    mplot = monthly.copy()
    mplot["month_str"] = mplot["month"].dt.strftime("%b-%Y")
    mplot = mplot.dropna(subset=["TIR %"])
    if not len(mplot):
        st.info("No valid monthly data to plot.")
    else:
        order = mplot["month_str"].tolist()
        subtitle = window_label
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

st.divider()
st.subheader("📖 Best Practice Targets")
with st.expander("View international consensus guidelines"):
    st.markdown("""
    These benchmarks are based on the **International Consensus on Time in Range (2019)**, widely adopted by ADA/EASD.

    - **Time in Range (3.9–10 mmol/L):** ≥ 70%  
    - **Time Above Range (10–13.9 mmol/L):** < 25%  
    - **Time Above Range (>13.9 mmol/L):** < 5%  
    - **Time Below Range (3.0–3.9 mmol/L):** < 4%  
    - **Time Below Range (<3.0):** < 1%  
    - **Mean SG:** roughly aligns to HbA1c target set with your clinician
    """)

st.caption(
    "Tables show 2 decimal places. Analysis window is set in the sidebar (3/6/12 months or All data). "
    "Benchmark colours: Green meets target • Amber near target • Red outside target."
)

# =========================
#        PDF EXPORT
# =========================

def _mini_tir_line_png(monthly_df: pd.DataFrame) -> str:
    """Return path to a PNG line plot of TIR % by month."""
    if not len(monthly_df): return ""
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=150)
    x = pd.to_datetime(monthly_df["month"].astype(str).str.replace(r'-\d+$','-01', regex=True))
    ax.plot(x, monthly_df["TIR %"], marker="o", linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("TIR %")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.tight_layout()
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _mini_mean_line_png(monthly_df: pd.DataFrame) -> str:
    """Return path to a PNG line plot of Mean SG by month."""
    if not len(monthly_df): return ""
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=150)
    x = pd.to_datetime(monthly_df["month"].astype(str).str.replace(r'-\d+$','-01', regex=True))
    ax.plot(x, monthly_df["Mean SG (mmol/L)"], marker="o", linewidth=1.5)
    ax.set_ylim(3, 15)
    ax.set_ylabel("Mean SG (mmol/L)")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.tight_layout()
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _hourly_tir_bar_png(hourly_df: pd.DataFrame) -> str:
    """Return path to a PNG bar chart of hourly Time in Range %."""
    if not len(hourly_df): return ""
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=150)
    ax.bar(hourly_df["hour"], hourly_df["Time in Range %"], width=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Hour")
    ax.set_ylabel("TIR %")
    ax.grid(True, axis="y", alpha=0.3)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.tight_layout()
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _generate_commentary(monthly_df: pd.DataFrame, hourly_df: pd.DataFrame) -> list[str]:
    """Very short bullet points based on simple rules."""
    notes = []
    if len(monthly_df) >= 2:
        last = monthly_df.iloc[-1]
        prev = monthly_df.iloc[-2]
        if pd.notna(last["TIR %"]) and pd.notna(prev["TIR %"]):
            delta = last["TIR %"] - prev["TIR %"]
            if abs(delta) >= 2:
                direction = "up" if delta > 0 else "down"
                notes.append(f"TIR {direction} {delta:.1f} pp vs previous month.")
        if pd.notna(last["Mean SG (mmol/L)"]) and pd.notna(prev["Mean SG (mmol/L)"]):
            d2 = last["Mean SG (mmol/L)"] - prev["Mean SG (mmol/L)"]
            if abs(d2) >= 0.2:
                direction = "higher" if d2 > 0 else "lower"
                notes.append(f"Mean SG {direction} by {abs(d2):.2f} mmol/L vs previous month.")
    # guideline flags
    if len(monthly_df):
        cur = monthly_df.iloc[-1]
        if pd.notna(cur.get("TBR % (<3.0)", np.nan)) and cur["TBR % (<3.0)"] > 1:
            notes.append("Time below 3.0 mmol/L above the 1% guideline — review overnight and pre-exercise lows.")
        if pd.notna(cur.get("TAR % (>13.9)", np.nan)) and cur["TAR % (>13.9)"] > 5:
            notes.append("Severe hyper (>13.9 mmol/L) above the 5% guideline — check meal timing/bolus strategy.")
        if pd.notna(cur.get("TIR %", np.nan)) and cur["TIR %"] < 70:
            notes.append("TIR below the ≥70% guideline — focus on post-meal control and basal alignment.")
    # hourly hotspots
    if len(hourly_df):
        worst_hour = hourly_df.sort_values("Time in Range %").iloc[0]
        if worst_hour["Time in Range %"] < 60:
            notes.append(f"Lowest TIR hour: {int(worst_hour['hour']):02d}:00 (~{worst_hour['Time in Range %']:.0f}%).")
    return notes[:4]  # keep it tight

def build_pdf(patient: str, window_label: str, monthly_df: pd.DataFrame, hourly_df: pd.DataFrame, include_comments: bool) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1*cm, rightMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Tiny", fontSize=8, leading=10))
    story = []

    title = Paragraph(f"<b>{patient} — Diabetes Summary</b>", styles["Title"])
    sub = Paragraph(f"<i>{window_label}</i>", styles["Normal"])
    story += [title, sub, Spacer(1, 6)]

    # Key metrics row (use last month if available)
    key_rows = []
    if len(monthly_df):
        last = monthly_df.iloc[-1]
        def fmt(v, pct=False): 
            if pd.isna(v): return "—"
            return f"{v:.2f}%" if pct else f"{v:.2f}"
        key_rows = [
            ["Mean SG (mmol/L)", fmt(last.get("Mean SG (mmol/L)"))],
            ["GMI (%)", fmt(last.get("GMI %"), pct=True)],
            ["TIR %", fmt(last.get("TIR %"), pct=True)],
            ["TBR % (<3.0)", fmt(last.get("TBR % (<3.0)"), pct=True)],
            ["TAR % (>13.9)", fmt(last.get("TAR % (>13.9)"), pct=True)],
        ]
    if key_rows:
        t = Table(key_rows, colWidths=[6*cm, 3*cm])
        t.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), "Helvetica", 9),
            ("ALIGN", (1,0), (1,-1), "RIGHT"),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ]))
        story += [t, Spacer(1, 6)]

    # Charts
    tir_img = _mini_tir_line_png(monthly_df) if len(monthly_df) else ""
    mean_img = _mini_mean_line_png(monthly_df) if len(monthly_df) else ""
    hourly_img = _hourly_tir_bar_png(hourly_df) if len(hourly_df) else ""

    img_w = 8.5*cm
    row = []
    if tir_img: row.append(Image(tir_img, width=img_w, height=3.3*cm))
    if mean_img: row.append(Image(mean_img, width=img_w, height=3.3*cm))
    if row:
        tbl = Table([[*row]])
        tbl.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
        story += [tbl, Spacer(1, 4)]
    if hourly_img:
        story += [Paragraph("<b>Hourly Time in Range</b>", styles["Normal"]),
                  Image(hourly_img, width=17.5*cm, height=4.2*cm),
                  Spacer(1, 6)]

    # Optional commentary
    if include_comments:
        notes = _generate_commentary(monthly_df, hourly_df)
        if notes:
            story += [Paragraph("<b>Notes</b>", styles["Normal"])]
            for n in notes:
                story += [Paragraph(f"• {n}", styles["Normal"])]
            story += [Spacer(1, 6)]

    # Footnote
    foot = Paragraph(
        "Benchmarks from International Consensus on Time in Range (2019): "
        "TIR ≥70%, TAR 10–13.9 <25%, TAR >13.9 <5%, TBR 3.0–3.9 <4%, TBR <3.0 <1%.",
        styles["Tiny"]
    )
    story += [Spacer(1, 4), foot]

    doc.build(story)
    return buf.getvalue()

# ---------- build hourly df for export ----------
hourly_export = pd.DataFrame()
if have_sg:
    tmp2 = analysis[["dt","SG"]].dropna().copy()
    tmp2["hour"] = tmp2["dt"].dt.hour
    hourly_export = tmp2.groupby("hour").agg(
        **{"Time in Range %": ("SG", lambda s: ((s>=3.9)&(s<=10.0)).mean()*100)}
    ).reset_index().round(2)

# ---------- export button ----------
st.subheader("Doctor export")

if not PDF_AVAILABLE:
    st.info(
        "PDF export is temporarily unavailable on this deployment. "
        "Install dependencies and redeploy: `matplotlib` and `reportlab`."
    )
else:
    col_a, _ = st.columns([1,3])
    with col_a:
        if st.button("🧾 Generate one-page PDF"):
            pdf_bytes = build_pdf(patient_name, window_label, monthly_display.copy(), hourly_export, include_comments)
            st.download_button("Download PDF", data=pdf_bytes, file_name="diabetes_summary.pdf", mime="application/pdf")
