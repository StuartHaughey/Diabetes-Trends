# app.py — Diabetes Trends (CareLink multi-section CSVs via GitHub)
# v7: adds cleaning + metrics + TIR by month + hourly profile

import os
import re
from urllib.parse import quote
from urllib.request import urlopen

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 0) Query params (keep-alive + debug)
# ──────────────────────────────────────────────────────────────
qp = st.experimental_get_query_params()
if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()
DEBUG = qp.get("debug") in (["1"], "1")

# ──────────────────────────────────────────────────────────────
# 1) Config
# ──────────────────────────────────────────────────────────────
GITHUB_USER = os.getenv("DATA_GITHUB_USER", "stuarthaughey")
DATA_REPO   = os.getenv("DATA_REPO_NAME", "diabetes-data")
BRANCH      = os.getenv("DATA_REPO_BRANCH", "main")
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"

SYD_TZ = ZoneInfo("Australia/Sydney")

# ──────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────
MONTH_PATTERN = re.compile(r"([A-Za-z]{3}-\d{4})\.csv$")  # e.g. "Sep-2025"

def extract_month_str(fname: str) -> str:
    m = MONTH_PATTERN.search(str(fname).strip())
    if not m:
        raise ValueError(f"Cannot find MMM-YYYY in filename: {fname}")
    return m.group(1)

def parse_month_to_date(mmm_yyyy: str) -> datetime:
    return datetime.strptime(mmm_yyyy, "%b-%Y")

@st.cache_resource
def get_status():
    return {"app_start": datetime.now(timezone.utc), "last_refresh": None, "version": "v7"}

def fmt_dt(dt_utc):
    if not dt_utc:
        return "—"
    return dt_utc.astimezone(SYD_TZ).strftime("%a %d %b %Y, %I:%M:%S %p AEST")

def human_delta(td: timedelta):
    s = int(td.total_seconds())
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60: return f"{m}m {s}s"
    h, m = divmod(m, 60)
    if h < 24: return f"{h}h {m}m"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"

HEADER_PREFIX = "Index,Date,Time"  # start of the real data table
GLUCOSE_CANDIDATES = [
    "Sensor Glucose (mmol/L)",
    "Sensor Glucose (mg/dL)",
    "SG (mmol/L)",
    "SG (mg/dL)",
]
DATE_COL = "Date"
TIME_COL = "Time"

# ──────────────────────────────────────────────────────────────
# 3) Data loaders
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_index() -> pd.DataFrame:
    url = RAW_BASE + "index.csv"
    idx = pd.read_csv(url)
    if "file" not in idx.columns:
        raise ValueError("index.csv must contain a 'file' column")
    files = [str(x).strip() for x in idx["file"].dropna().tolist()]
    rows = []
    for f in files:
        mstr = extract_month_str(f)
        mdate = parse_month_to_date(mstr)
        rows.append({"file": f, "month": mstr, "month_date": mdate})
    return pd.DataFrame(rows).sort_values("month_date").reset_index(drop=True)

def _first_300_lines(url: str) -> list[str]:
    with urlopen(url) as resp:
        text = resp.read().decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    return lines[:300]

def _find_header_row(url: str) -> int:
    lines = _first_300_lines(url)
    for i, line in enumerate(lines):
        if line.startswith(HEADER_PREFIX):
            return i
    for i, line in enumerate(lines):
        if "Date" in line and "Time" in line and line.startswith("Index"):
            return i
    return 0

@st.cache_data(ttl=3600)
def load_month_csv(filename: str) -> pd.DataFrame:
    """Load a single month, reading only the time-series table."""
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)
    if DEBUG: st.code(f"Fetching: {url}")

    header_row = _find_header_row(url)

    df = pd.read_csv(
        url,
        skiprows=header_row,
        header=0,
        sep=",",
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip",
        quotechar='"'
    )

    # Clean up
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # Keep rows that have Date & Time
    if DATE_COL in df.columns and TIME_COL in df.columns:
        df = df[df[DATE_COL].notna() & df[TIME_COL].notna()]

    # Month label
    df["month"] = extract_month_str(clean)

    # Datetime (AEST)
    if DATE_COL in df.columns and TIME_COL in df.columns:
        try:
            dt = pd.to_datetime(
                df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
                errors="coerce",
                dayfirst=True,
            )
            df["dt"] = dt.dt.tz_localize(SYD_TZ, nonexistent="NaT", ambiguous="NaT")
        except Exception:
            df["dt"] = pd.NaT
    else:
        df["dt"] = pd.NaT

    if DEBUG:
        st.write(f"Header row detected at line: {header_row}")
        st.write("Columns:", list(df.columns)[:16])
        st.dataframe(df.head(12))

    return df

@st.cache_data(ttl=3600)
def load_all_months(meta: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in meta["file"].tolist():
        try:
            frames.append(load_month_csv(f))
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 4) Cleaning + metrics
# ──────────────────────────────────────────────────────────────
def pick_glucose_column(df: pd.DataFrame) -> str | None:
    for g in GLUCOSE_CANDIDATES:
        if g in df.columns:
            return g
    return None

def clean_glucose(df: pd.DataFrame, glu_col: str) -> pd.DataFrame:
    out = df.copy()
    # Convert common "None" strings to NaN then numeric
    out[glu_col] = pd.to_numeric(out[glu_col].replace("None", pd.NA), errors="coerce")
    # Drop rows with no timestamp or no glucose
    if "dt" in out.columns:
        out = out[out["dt"].notna()]
    out = out[out[glu_col].notna()]
    # Keep obviously valid physiologic range for mmol/L/mg/dL
    if "mmol/L" in glu_col:
        out = out[(out[glu_col] >= 1.5) & (out[glu_col] <= 33)]
    else:
        # mg/dL
        out = out[(out[glu_col] >= 30) & (out[glu_col] <= 600)]
        # convert to mmol/L for unified analysis
        out["glucose_mmol"] = out[glu_col] / 18.0
        return out.rename(columns={"glucose_mmol": "glucose"})
    out = out.rename(columns={glu_col: "glucose"})
    return out

def compute_metrics(df: pd.DataFrame, lower: float, upper: float) -> dict:
    if df.empty or "glucose" not in df.columns:
        return {}
    g = df["glucose"]
    n = len(g)
    mean = g.mean()
    median = g.median()
    tir = (g.between(lower, upper)).mean() * 100
    low = (g < lower).mean() * 100
    high = (g > upper).mean() * 100
    # HbA1c estimate (NGSP): HbA1c% = (eAG_mgdl + 46.7) / 28.7 ; eAG_mgdl = mean_mmol * 18
    eAG_mgdl = mean * 18.0
    hba1c_pct = (eAG_mgdl + 46.7) / 28.7
    # IFCC mmol/mol (approx): (HbA1c% - 2.15) * 10.929
    hba1c_ifcc = (hba1c_pct - 2.15) * 10.929
    return dict(
        samples=n,
        mean_mmol=mean,
        median_mmol=median,
        tir_pct=tir,
        low_pct=low,
        high_pct=high,
        eAG_mgdl=eAG_mgdl,
        hba1c_pct=hba1c_pct,
        hba1c_ifcc=hba1c_ifcc,
    )

def monthly_tir(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    if df.empty or "glucose" not in df.columns:
        return pd.DataFrame(columns=["month","TIR %"])
    grp = df.groupby("month")["glucose"]
    tir = (grp.apply(lambda s: s.between(lower, upper).mean() * 100)).reset_index(name="TIR %")
    return tir

def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dt" not in df.columns or "glucose" not in df.columns:
        return pd.DataFrame(columns=["hour","mean_mmol"])
    temp = df.copy()
    temp["hour"] = temp["dt"].dt.hour
    prof = temp.groupby("hour")["glucose"].mean().reset_index(name="mean_mmol")
    return prof

# ──────────────────────────────────────────────────────────────
# 5) Sidebar + UI
# ──────────────────────────────────────────────────────────────
def render_status_sidebar():
    s = get_status()
    now = datetime.now(timezone.utc)
    st.sidebar.markdown("### App Status")
    st.sidebar.write(f"Start: {fmt_dt(s['app_start'])}")
    st.sidebar.write(f"Uptime: {human_delta(now - s['app_start'])}")
    st.sidebar.write(f"Last refresh: {fmt_dt(s['last_refresh'])}")
    st.sidebar.caption("Health `?ping=1` • Debug `?debug=1`")

    st.sidebar.markdown("### Targets (mmol/L)")
    lower = st.sidebar.number_input("Lower", value=3.9, step=0.1, format="%.1f")
    upper = st.sidebar.number_input("Upper", value=10.0, step=0.1, format="%.1f")
    return lower, upper

def month_selector(meta: pd.DataFrame):
    months = meta["month"].tolist()
    latest = months[-1]
    mode = st.sidebar.radio("View", ["Latest", "Single", "All"], index=0)
    if mode == "Latest":
        return "latest", latest
    elif mode == "Single":
        sel = st.sidebar.selectbox("Choose month", months, index=len(months)-1)
        return "single", sel
    else:
        return "all", None

# ──────────────────────────────────────────────────────────────
# 6) Main app
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")

lower_target, upper_target = render_status_sidebar()
meta = load_index()

view_mode, sel_month = month_selector(meta)
if view_mode == "all":
    raw = load_all_months(meta)
elif view_mode == "single":
    fname = meta.loc[meta["month"] == sel_month, "file"].iloc[0]
    raw = load_month_csv(fname)
else:
    latest = meta["month"].iloc[-1]
    fname = meta.loc[meta["month"] == latest, "file"].iloc[0]
    raw = load_month_csv(fname)

get_status()["last_refresh"] = datetime.now(timezone.utc)

# Pick & clean glucose
GLU = pick_glucose_column(raw)
clean = clean_glucose(raw, GLU) if GLU else pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 7) Summary + metrics
# ──────────────────────────────────────────────────────────────
st.subheader("Summary")
st.write(f"Months: **{len(meta)}**, Range: **{meta['month'].iloc[0]} → {meta['month'].iloc[-1]}**")
st.write(f"Rows loaded (raw): **{len(raw)}** — Clean rows: **{len(clean)}**")
if GLU:
    st.caption(f"Glucose column detected: `{GLU}` (converted to mmol/L as `glucose`)")

# Metrics
metrics = compute_metrics(clean, lower_target, upper_target)
if metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean (mmol/L)", f"{metrics['mean_mmol']:.1f}")
    c2.metric("Median (mmol/L)", f"{metrics['median_mmol']:.1f}")
    c3.metric("Time In Range", f"{metrics['tir_pct']:.0f}%")
    c4.metric("Low", f"{metrics['low_pct']:.0f}%")
    c5.metric("High", f"{metrics['high_pct']:.0f}%")
    c6, c7 = st.columns(2)
    c6.metric("eAG (mg/dL)", f"{metrics['eAG_mgdl']:.0f}")
    c7.metric("HbA1c (NGSP % / IFCC mmol/mol)",
              f"{metrics['hba1c_pct']:.1f}%  /  {metrics['hba1c_ifcc']:.0f}")

st.divider()
st.subheader("Raw preview (first 200 rows)")
st.dataframe(raw.head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 8) Charts
# ──────────────────────────────────────────────────────────────
if not clean.empty and "dt" in clean.columns:
    st.subheader("Glucose over time")
    st.line_chart(clean[["dt", "glucose"]].set_index("dt"))

    st.subheader("Hourly profile (mean mmol/L)")
    prof = hourly_profile(clean)
    if not prof.empty:
        st.bar_chart(prof.set_index("hour")["mean_mmol"])

    st.subheader("Monthly Time-in-Range (%)")
    tir = monthly_tir(clean, lower_target, upper_target)
    if not tir.empty:
        st.bar_chart(tir.set_index("month")["TIR %"])
else:
    st.info("No cleaned glucose rows yet — check the detected glucose column or targets.")
