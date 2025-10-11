# app.py — Diabetes Trends (robust version)
# Connects to a GitHub data repo containing:
#   index.csv (column: file)
#   Stuart Haughey Jan-2025.csv, Stuart Haughey Feb-2025.csv, etc.

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0) Fast path for keep-alive pings
# ──────────────────────────────────────────────────────────────────────────────
qp = getattr(st, "query_params", None)
if qp is None:
    qp = st.experimental_get_query_params()

if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()

DEBUG = qp.get("debug") in (["1"], "1")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Config
# ──────────────────────────────────────────────────────────────────────────────
GITHUB_USER = "stuarthaughey"     # your GitHub username
DATA_REPO   = "diabetes-data"     # name of your data repo
BRANCH      = "main"              # usually 'main'
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"

SYD_TZ = ZoneInfo("Australia/Sydney")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Utility functions
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_status():
    return {
        "app_start": datetime.now(timezone.utc),
        "last_refresh": None,
    }

def fmt_dt(dt_utc):
    if not dt_utc:
        return "—"
    return dt_utc.astimezone(SYD_TZ).strftime("%a %d %b %Y, %I:%M:%S %p AEST")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Data loading
# ──────────────────────────────────────────────────────────────────────────────
MONTH_PATTERN = re.compile(r"([A-Za-z]{3}-\d{4})\.csv$")

def extract_month(fname: str) -> str:
    m = MONTH_PATTERN.search(str(fname).strip())
    if not m:
        raise ValueError(f"Cannot find MMM-YYYY in filename: {fname}")
    return m.group(1)

def parse_month(mmm_yyyy: str) -> datetime:
    return datetime.strptime(mmm_yyyy, "%b-%Y")

@st.cache_data(ttl=3600)
def load_index():
    url = RAW_BASE + "index.csv"
    idx = pd.read_csv(url)
    files = [str(x).strip() for x in idx["file"].dropna().tolist()]
    meta = []
    for f in files:
        m = extract_month(f)
        meta.append({"file": f, "month": m, "month_date": parse_month(m)})
    meta = pd.DataFrame(meta).sort_values("month_date").reset_index(drop=True)
    return meta

@st.cache_data(ttl=3600)
def load_month_csv(fname: str) -> pd.DataFrame:
    """Fetch one month’s CSV, tolerate inconsistent formatting."""
    url = RAW_BASE + quote(fname.strip())
    if DEBUG:
        st.code(f"Fetching: {url}")

    attempts = [
        dict(sep=None, engine="python"),
        dict(sep=",", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep=";", engine="python"),
        dict(sep=None, engine="python", encoding="utf-8-sig"),
    ]

    df = None
    used_opts = None
    last_err = None

    for opts in attempts:
        try:
            df = pd.read_csv(url, on_bad_lines="skip", **opts)
            if df.shape[1] == 1 or all(str(c).startswith("Unnamed") for c in df.columns):
                raise ValueError("delimiter guess failed")
            used_opts = opts
            break
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        st.error(f"Failed to parse file: {fname}")
        st.code(url)
        st.exception(last_err)
        raise RuntimeError(f"Could not parse {fname}")

    if DEBUG:
        st.write("Detected delimiter:", used_opts.get("sep") if used_opts else "?")
        st.write("Columns detected:", list(df.columns))
        st.dataframe(df.head())

    df["month"] = extract_month(fname)
    return df

@st.cache_data(ttl=3600)
def load_all_months(meta: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in meta["file"]:
        try:
            frames.append(load_month_csv(f))
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def sidebar_status():
    s = get_status()
    st.sidebar.markdown("### App Status")
    st.sidebar.write(f"Start: {fmt_dt(s['app_start'])}")
    st.sidebar.write(f"Last refresh: {fmt_dt(s['last_refresh'])}")
    st.sidebar.caption("Health: ?ping=1 • Debug: ?debug=1")

def month_selector(meta: pd.DataFrame):
    months = meta["month"].tolist()
    latest = months[-1]
    mode = st.sidebar.radio("View", ["Latest", "Single", "All"], index=0)
    if mode == "Single":
        sel = st.sidebar.selectbox("Choose month", months, index=len(months)-1)
        return "single", sel
    elif mode == "All":
        return "all", None
    else:
        return "latest", latest

# ──────────────────────────────────────────────────────────────────────────────
# 5) Main app
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")

sidebar_status()

with st.spinner("Loading index..."):
    meta = load_index()

if DEBUG:
    st.info("Raw URLs being fetched:")
    for f in meta["file"]:
        st.code(RAW_BASE + quote(str(f).strip()))

mode, sel = month_selector(meta)

if mode == "all":
    with st.spinner("Loading all months..."):
        data = load_all_months(meta)
elif mode == "single":
    with st.spinner(f"Loading {sel}..."):
        fname = meta.loc[meta["month"] == sel, "file"].iloc[0]
        data = load_month_csv(fname)
else:
    latest = meta["month"].iloc[-1]
    with st.spinner(f"Loading {latest}..."):
        fname = meta.loc[meta["month"] == latest, "file"].iloc[0]
        data = load_month_csv(fname)

get_status()["last_refresh"] = datetime.now(timezone.utc)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Display
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Summary")
st.write(f"Months: {len(meta)}  |  Rows: {len(data)}")
st.write(f"Range: {meta['month'].iloc[0]} → {meta['month'].iloc[-1]}")

st.divider()
st.subheader("Data preview")
st.dataframe(data.head(200), use_container_width=True)

if not data.empty:
    numeric_cols = [c for c in data.select_dtypes("number").columns if c != "month"]
    if numeric_cols:
        col = st.selectbox("Numeric column for monthly mean:", numeric_cols, index=0)
        means = data.groupby("month", as_index=False)[col].mean()
        st.bar_chart(means.set_index("month")[col])
    else:
        st.caption("No numeric columns detected yet — check data format.")

st.divider()
st.caption("Tip: add '?debug=1' to the URL to see exact files being fetched.")
