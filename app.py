# app.py — Diabetes Trends (CareLink multi-section CSVs via GitHub)

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 0) Keep-alive fast path
# ──────────────────────────────────────────────────────────────
qp = getattr(st, "query_params", None) or st.experimental_get_query_params()
if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()
DEBUG = qp.get("debug") in (["1"], "1")

# ──────────────────────────────────────────────────────────────
# 1) Config
# ──────────────────────────────────────────────────────────────
GITHUB_USER = "stuarthaughey"
DATA_REPO   = "diabetes-data"
BRANCH      = "main"
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"
SYD_TZ      = ZoneInfo("Australia/Sydney")

# ──────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────
MONTH_PATTERN = re.compile(r"([A-Za-z]{3}-\d{4})\.csv$")

def extract_month_str(fname: str) -> str:
    m = MONTH_PATTERN.search(str(fname).strip())
    if not m:
        raise ValueError(f"Cannot find MMM-YYYY in filename: {fname}")
    return m.group(1)

def parse_month_to_date(mmm_yyyy: str) -> datetime:
    return datetime.strptime(mmm_yyyy, "%b-%Y")

@st.cache_resource
def get_status():
    return {"app_start": datetime.now(timezone.utc), "last_refresh": None, "version": "v2"}

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

# ──────────────────────────────────────────────────────────────
# 3) Data loaders
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_index() -> pd.DataFrame:
    url = RAW_BASE + "index.csv"
    idx = pd.read_csv(url)
    files = [str(x).strip() for x in idx["file"].dropna().tolist()]
    rows = []
    for f in files:
        mstr = extract_month_str(f)
        mdate = parse_month_to_date(mstr)
        rows.append({"file": f, "month": mstr, "month_date": mdate})
    meta = pd.DataFrame(rows).sort_values("month_date").reset_index(drop=True)
    return meta

# Columns we care about (CareLink varies slightly by export)
GLUCOSE_CANDIDATES = [
    "Sensor Glucose (mmol/L)",
    "Sensor Glucose (mg/dL)",
    "SG (mmol/L)",
    "SG (mg/dL)",
]
DATE_COL = "Date"
TIME_COL = "Time"

def _find_timeseries_header_row(url: str) -> int:
    """
    Scan the first ~200 lines to find the header of the time-series/events table.
    Prefer a row that begins with 'Index,Date,Time' and also contains a glucose column.
    """
    sample = pd.read_csv(
        url, header=None, sep=",", engine="python",
        nrows=200, on_bad_lines="skip", encoding="utf-8-sig"
    )

    # First pass: exact "Index,Date,Time" AND a glucose candidate present
    for i, row in sample.iterrows():
        cells = [str(x).strip() for x in row.tolist()]
        if len(cells) >= 3 and cells[0] == "Index" and cells[1] == "Date" and cells[2] == "Time":
            if any(g in cells for g in GLUCOSE_CANDIDATES):
                return i

    # Second pass: any row containing Date + Time + a glucose column
    for i, row in sample.iterrows():
        cells = [str(x).strip() for x in row.tolist()]
        if DATE_COL in cells and TIME_COL in cells and any(g in cells for g in GLUCOSE_CANDIDATES):
            return i

    # Fallback: densest row (most non-nulls)
    return sample.count(axis=1).idxmax()

@st.cache_data(ttl=3600)
def load_month_csv(filename: str) -> pd.DataFrame:
    """Robust loader for CareLink multi-section CSVs → returns time-series rows."""
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)
    if DEBUG: st.code(f"Fetching: {url}")

    header_row = _find_timeseries_header_row(url)

    df = pd.read_csv(
        url, sep=",", engine="python", header=header_row,
        on_bad_lines="skip", encoding="utf-8-sig", quotechar='"'
    )
    # Drop completely empty columns, trim names
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # Keep only rows that look like real time entries: Date & Time present
    if DATE_COL in df.columns and TIME_COL in df.columns:
        df = df[df[DATE_COL].notna() & df[TIME_COL].notna()]

    # Add month label
    df["month"] = extract_month_str(clean)

    # Build a proper datetime column (AEST)
    try:
        # Some exports: Date = 29/09/2025, Time = 21:30:05
        dt = pd.to_datetime(df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str), errors="coerce", dayfirst=True)
        df["dt"] = dt.dt.tz_localize(SYD_TZ, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        df["dt"] = pd.NaT

    if DEBUG:
        st.write(f"Header row detected: {header_row}")
        st.write("Columns:", list(df.columns)[:14])
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
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    get_status()["last_refresh"] = datetime.now(timezone.utc)
    return df

# ──────────────────────────────────────────────────────────────
# 4) Sidebar UI
# ──────────────────────────────────────────────────────────────
def render_status_sidebar():
    s = get_status()
    now = datetime.now(timezone.utc)
    uptime = human_delta(now - s["app_start"])
    st.sidebar.markdown("### App Status")
    st.sidebar.write(f"**Start:** {fmt_dt(s['app_start'])}")
    st.sidebar.write(f"**Uptime:** {uptime}")
    st.sidebar.write(f"**Last refresh:** {fmt_dt(s['last_refresh'])}")
    st.sidebar.caption("Health `?ping=1` • Debug `?debug=1`")

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
# 5) App body
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")

render_status_sidebar()

with st.spinner("Loading index…"):
    meta = load_index()

if DEBUG:
    st.info("Constructed RAW URLs:")
    for f in meta["file"]:
        st.code(RAW_BASE + quote(f.strip()))

view_mode, sel_month = month_selector(meta)

if view_mode == "all":
    with st.spinner("Loading all months…"):
        data = load_all_months(meta)
elif view_mode == "single":
    with st.spinner(f"Loading {sel_month}…"):
        fname = meta.loc[meta["month"] == sel_month, "file"].iloc[0]
        data = load_month_csv(fname)
        get_status()["last_refresh"] = datetime.now(timezone.utc)
else:
    latest = meta["month"].iloc[-1]
    with st.spinner(f"Loading {latest}…"):
        fname = meta.loc[meta["month"] == latest, "file"].iloc[0]
        data = load_month_csv(fname)
        get_status()["last_refresh"] = datetime.now(timezone.utc)

# ──────────────────────────────────────────────────────────────
# 6) Summary + preview
# ──────────────────────────────────────────────────────────────
st.subheader("Summary")
st.write(f"Months: **{len(meta)}**, Range: {meta['month'].iloc[0]} → {meta['month'].iloc[-1]}")
st.write(f"Rows loaded: **{len(data)}**")

csv_bytes = data.to_csv(index=False).encode("utf-8")
st.download_button("Download current view (CSV)", csv_bytes,
                   file_name="diabetes_trends.csv", mime="text/csv")

st.divider()
st.subheader("Raw preview (first 200 rows)")
st.dataframe(data.head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 7) Minimal analysis scaffolding (will improve once columns confirmed)
# ──────────────────────────────────────────────────────────────
if not data.empty:
    # Choose the first glucose column we can find
    glucose_col = next((g for g in GLUCOSE_CANDIDATES if g in data.columns), None)

    # Rows per month
    counts = data.groupby("month", as_index=False).size().rename(columns={"size": "rows"})
    counts = counts.merge(meta[["month"]], on="month", how="right")
    st.write("Rows per month")
    st.bar_chart(counts.set_index("month")["rows"])

    # Quick monthly mean if a glucose column exists
    if glucose_col:
        st.write(f"Monthly mean of **{glucose_col}**")
        means = data.groupby("month", as_index=False)[glucose_col].mean(numeric_only=True)
        means = means.merge(meta[["month"]], on="month", how="right")
        st.line_chart(means.set_index("month")[glucose_col])
    else:
        st.caption("Glucose column not found yet — once we confirm the exact header name, I’ll wire proper charts.")
else:
    st.info("No time-series rows detected. If preview still shows patient/device tables, we’ll tweak the header finder.")
