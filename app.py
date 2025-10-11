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

@st.cache_data(ttl=3600, show_spinner=False)
def load_month_csv(filename: str) -> pd.DataFrame:
    """
    Robust loader for CareLink/780G exports.
    - Finds the real header row (starts with 'Index,Date,Time').
    - Reads with comma separator, skips malformed lines.
    """
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)

    if DEBUG:
        st.code(f"Fetching: {url}")

    # 1) Peek the first 100 lines to locate the true header row
    sample = pd.read_csv(
        url,
        header=None,
        sep=",",
        engine="python",
        nrows=100,
        on_bad_lines="skip",
        encoding="utf-8-sig",
    )

    # header row is the first row whose first 3 cells match 'Index','Date','Time'
    header_row = None
    for i, row in sample.iterrows():
        # be defensive about NA/None
        first = str(row.iloc[0]).strip() if len(row) > 0 else ""
        second = str(row.iloc[1]).strip() if len(row) > 1 else ""
        third = str(row.iloc[2]).strip() if len(row) > 2 else ""
        if first == "Index" and second == "Date" and third == "Time":
            header_row = i
            break

    if header_row is None:
        # Fallback: pick the row with the most non-null entries
        header_row = sample.count(axis=1).idxmax()

    # 2) Read full file using that header row
    df = pd.read_csv(
        url,
        sep=",",
        engine="python",
        header=header_row,
        on_bad_lines="skip",
        encoding="utf-8-sig",
        quotechar='"'
    )

    # 3) Basic tidy-up
    # drop completely-empty columns
    df = df.dropna(axis=1, how="all")
    # strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # 4) Tag month from filename
    df["month"] = extract_month_str(clean)

    if DEBUG:
        st.write(f"Header row detected: {header_row}")
        st.write("Columns:", list(df.columns)[:12])
        st.dataframe(df.head(10))

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
