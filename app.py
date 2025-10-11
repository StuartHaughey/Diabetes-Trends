# app.py — Diabetes Trends (GitHub-hosted monthly CSVs)
# Data repo layout (public):
#   diabetes-data/
#     index.csv                # column: file
#     Stuart Haughey Jan-2025.csv
#     Stuart Haughey Feb-2025.csv
#     ...

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0) Fast path for keep-alive pings (must be at the top)
#    e.g. https://<your-app>.streamlit.app/?ping=1
# ──────────────────────────────────────────────────────────────────────────────
qp = getattr(st, "query_params", None)
if qp is None:  # older Streamlit fallback
    qp = st.experimental_get_query_params()

if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()

DEBUG = qp.get("debug") in (["1"], "1")

# ──────────────────────────────────────────────────────────────────────────────
# 1) Config — set your GitHub data repo here
# ──────────────────────────────────────────────────────────────────────────────
GITHUB_USER = os.getenv("DATA_GITHUB_USER", "stuarthaughey")
DATA_REPO   = os.getenv("DATA_REPO_NAME", "diabetes-data")
BRANCH      = os.getenv("DATA_REPO_BRANCH", "main")
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"

SYD_TZ      = ZoneInfo("Australia/Sydney")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Tiny status store (persists across reruns in a session)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_status():
    return {
        "app_start": datetime.now(timezone.utc),
        "last_data_refresh": None,
        "version": os.getenv("GIT_COMMIT", "dev"),
    }

def fmt_dt(dt_utc):
    if not dt_utc:
        return "—"
    return dt_utc.astimezone(SYD_TZ).strftime("%a %d %b %Y, %I:%M:%S %p AEST")

def human_delta(td: timedelta):
    secs = int(td.total_seconds())
    if secs < 60: return f"{secs}s"
    mins, s = divmod(secs, 60)
    if mins < 60: return f"{mins}m {s}s"
    hrs, m = divmod(mins, 60)
    if hrs < 24: return f"{hrs}h {m}m"
    days, h = divmod(hrs, 24)
    return f"{days}d {h}h"

# ──────────────────────────────────────────────────────────────────────────────
# 3) Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────
MONTH_PATTERN = re.compile(r"([A-Za-z]{3}-\d{4})\.csv$")  # e.g., "Sep-2025"

def extract_month_str(fname: str) -> str:
    m = MONTH_PATTERN.search(str(fname).strip())
    if not m:
        raise ValueError(f"Cannot find MMM-YYYY in filename: {fname}")
    return m.group(1)

def parse_month_to_date(mmm_yyyy: str) -> datetime:
    return datetime.strptime(mmm_yyyy, "%b-%Y")

@st.cache_data(ttl=3600, show_spinner=False)
def load_index() -> pd.DataFrame:
    """Load index.csv from the data repo. Expects a single column 'file'."""
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

@st.cache_data(ttl=3600, show_spinner=False)
def load_month_csv(filename: str) -> pd.DataFrame:
    """Fetch a single month's file and parse robustly (CSV/TSV/semicolon + BOM; header sniffing)."""
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)

    # Optional debug: show the exact URL we’re fetching
    if DEBUG:
        st.code(f"GET {url}")

    attempts = [
        dict(sep=None, engine="python"),                         # sniff delimiter
        dict(sep=",", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep=";", engine="python"),
        dict(sep=None, engine="python", encoding="utf-8-sig"),   # BOM
    ]

    last_err = None
    df = None
    for opts in attempts:
        try:
            df = pd.read_csv(url, on_bad_lines="skip", **opts)
            # if only one unnamed column, delimiter guess failed
            if df.shape[1] == 1 or all(str(c).startswith("Unnamed") for c in df.columns):
                raise ValueError("Likely wrong delimiter or header")
            break
        except Exception as e:
            last_err = e
            df = None

    # Try header sniff (files with a few lines before the real header)
    if df is None:
        try:
            sample = pd.read_csv(url, header=None, sep=None, engine="python", nrows=50)
            header_row = sample.count(axis=1).idxmax()
            df = pd.read_csv(url, sep=None, engine="python", header=header_row)
        except Exception as e:
            last_err = e
            df = None

    if df is None:
        st.error(f"Failed to parse file: {clean}")
        st.code(url)
        if last_err is not None:
            st.exception(last_err)
        raise RuntimeError(f"Could not parse {clean}")

    df["month"] = extract_month_str(clean)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_months(meta: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for fname in meta["file"].tolist():
        try:
            frames.append(load_month_csv(fname))
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    get_status()["last_data_refresh"] = datetime.now(timezone.utc)
    return all_df

st.write("Detected delimiter:", opts.get("sep"))
st.write("Columns detected:", list(df.columns))
st.dataframe(df.head(10))

# ──────────────────────────────────────────────────────────────────────────────
# 4) Sidebar — status + month selector
# ──────────────────────────────────────────────────────────────────────────────
def render_status_sidebar():
    s = get_status()
    now = datetime.now(timezone.utc)
    uptime = human_delta(now - s["app_start"])
    st.sidebar.markdown("### App Status")
    st.sidebar.write(f"**App start:** {fmt_dt(s['app_start'])}")
    st.sidebar.write(f"**Uptime:** {uptime}")
    st.sidebar.write(f"**Last data refresh:** {fmt_dt(s['last_data_refresh'])}")
    st.sidebar.write(f"**Version:** `{s['version']}`")
    st.sidebar.caption("Health URL: `?ping=1`  •  Debug: `?debug=1`")

def month_selector(meta: pd.DataFrame):
    if meta.empty:
        st.stop()
    months = meta["month"].tolist()
    latest = months[-1]
    mode = st.sidebar.radio("View", ["Latest month", "Single month", "All months"], index=0)
    if mode == "Latest month":
        return "latest", latest
    elif mode == "Single month":
        sel = st.sidebar.selectbox("Choose month", months, index=len(months)-1)
        return "single", sel
    else:
        return "all", None

# ──────────────────────────────────────────────────────────────────────────────
# 5) App body
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")

render_status_sidebar()

with st.spinner("Loading index…"):
    meta = load_index()

if DEBUG:
    st.info("Constructed RAW URLs:")
    for f in meta["file"]:
        st.code(RAW_BASE + quote(str(f).strip()))

view_mode, sel_month = month_selector(meta)

# Decide which data to load
if view_mode == "all":
    with st.spinner("Loading all months…"):
        data = load_all_months(meta)
elif view_mode == "single":
    with st.spinner(f"Loading {sel_month}…"):
        fname = meta.loc[meta["month"] == sel_month, "file"].iloc[0]
        data = load_month_csv(fname)
        get_status()["last_data_refresh"] = datetime.now(timezone.utc)
else:  # latest
    latest = meta["month"].iloc[-1]
    with st.spinner(f"Loading {latest}…"):
        fname = meta.loc[meta["month"] == latest, "file"].iloc[0]
        data = load_month_csv(fname)
        get_status()["last_data_refresh"] = datetime.now(timezone.utc)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Basic summary + download
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Summary")
    st.write(f"Months available: **{len(meta)}** — range: **{meta['month'].iloc[0]} → {meta['month'].iloc[-1]}**")
    st.write(f"Rows loaded: **{len(data)}**")
with right:
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download current view (CSV)", csv_bytes, file_name="diabetes_trends.csv", mime="text/csv")

st.divider()
st.subheader("Raw preview")
st.dataframe(data.head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 7) Analysis scaffolding — replace with your domain charts once columns known
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Analysis")

if data.empty:
    st.info("No data loaded. Check index.csv and filenames.")
else:
    # 1) Rows per month (requires 'month' column, which we set)
    counts = data.groupby("month", as_index=False).size().rename(columns={"size": "rows"})
    # Keep chronological order
    counts = counts.merge(meta[["month"]], on="month", how="right")
    st.write("Rows per month")
    st.bar_chart(counts.set_index("month")["rows"])

    # 2) If you have numeric columns, show a quick monthly mean
    numeric_cols = [c for c in data.select_dtypes("number").columns if c != "month"]
    if numeric_cols:
        col_to_show = st.selectbox("Numeric column for monthly mean:", numeric_cols, index=0)
        means = data.groupby("month", as_index=False)[col_to_show].mean()
        means = means.merge(meta[["month"]], on="month", how="right")
        st.write(f"Monthly mean of **{col_to_show}**")
        st.line_chart(means.set_index("month")[col_to_show])
    else:
        st.caption("When you share the real column names (e.g., glucose_mmol), I’ll wire proper diabetes charts.")
