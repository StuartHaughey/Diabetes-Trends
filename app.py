# app.py — Diabetes Trends (CareLink multi-section CSVs via GitHub)

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 0) Query params (use ONLY the experimental API to avoid conflicts)
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
    return {"app_start": datetime.now(timezone.utc), "last_refresh": None, "version": "v3"}

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
    if "file" not in idx.columns:
        raise ValueError("index.csv must contain a 'file' column")
    files = [str(x).strip() for x in idx["file"].dropna().tolist()]
    rows = []
    for f in files:
        mstr = extract_month_str(f)
        mdate = parse_month_to_date(mstr)
        rows.append({"file": f, "month": mstr, "month_date": mdate})
    meta = pd.DataFrame(rows).sort_values("month_date").reset_index(drop=True)
    return meta

GLUCOSE_CANDIDATES = [
    "Sensor Glucose (mmol/L)",
    "Sensor Glucose (mg/dL)",
    "SG (mmol/L)",
    "SG (mg/dL)",
]
DATE_COL = "Date"
TIME_COL = "Time"

def _find_timeseries_header_row(url: str) -> int:
    # Scan first 200 lines to locate the time-series header
    sample = pd.read_csv(
        url, header=None, sep=",", engine="python",
        nrows=200, on_bad_lines="skip", encoding="utf-8-sig"
    )
    # Best: exact "Index,Date,Time" + a glucose column
    for i, row in sample.iterrows():
        cells = [str(x).strip() for x in row.tolist()]
        if len(cells) >= 3 and cells[0] == "Index" and cells[1] == "Date" and cells[2] == "Time":
            if any(g in cells for g in GLUCOSE_CANDIDATES):
                return i
    # Next best: any row with Date + Time + a glucose column
    for i, row in sample.iterrows():
        cells = [str(x).strip() for x in row.tolist()]
        if DATE_COL in cells and TIME_COL in cells and any(g in cells for g in GLUCOSE_CANDIDATES):
            return i
    # Fallback: densest row
    return sample.count(axis=1).idxmax()

@st.cache_data(ttl=3600)
def load_month_csv(filename: str) -> pd.DataFrame:
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)
    if DEBUG: st.code(f"Fetching: {url}")

    header_row = _find_timeseries_header_row(url)

    df = pd.read_csv(
        url, sep=",", engine="python", header=header_row,
        on_bad_lines="skip", encoding="utf-8-sig", quotechar='"'
    )
    # Tidy columns
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # Keep rows that look like time entries
    if DATE_COL in df.columns and TIME_COL in df.columns:
        df = df[df[DATE_COL].notna() & df[TIME_COL].notna()]

    # Month label
    df["month"] = extract_month_str(clean)

    # Datetime (AEST)
    try:
        dt = pd.to_datetime(
            df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
            errors="coerce", dayfirst=True
        )
        df["dt"] = dt.dt.tz_localize(SYD_TZ, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        df["dt"] = pd.NaT

    if DEBUG:
        st.write(f"Header row detected: {header_row}")
        st.write("Columns (first 14):", list(df.columns)[:14])
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
# 7) Minimal analysis scaffolding
# ──────────────────────────────────────────────────────────────
if not data.empty:
    GLU = next((g for g in GLUCOSE_CANDIDATES if g in data.columns), None)

    counts = data.groupby("month", as_index=False).size().rename(columns={"size": "rows"})
    counts = counts.merge(meta[["month"]], on="month", how="right")
    st.write("Rows per month")
    st.bar_chart(counts.set_index("month")["rows"])

    if GLU:
        st.write(f"Monthly mean of **{GLU}**")
        means = data.groupby("month", as_index=False)[GLU].mean(numeric_only=True)
        means = means.merge(meta[["month"]], on="month", how="right")
        st.line_chart(means.set_index("month")[GLU])
    else:
        st.caption("Glucose column not found yet — once we confirm the exact header name, I’ll wire proper charts.")
else:
    st.info("No time-series rows detected. If preview still shows patient/device tables, we’ll tweak the header finder.")
