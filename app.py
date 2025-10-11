# app.py — Diabetes Trends (CareLink 780G Time-Series Parser)

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# 0. Keep-alive + debug
# ──────────────────────────────────────────────────────────────
qp = st.experimental_get_query_params()
if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()
DEBUG = qp.get("debug") in (["1"], "1")

# ──────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────
GITHUB_USER = "stuarthaughey"
DATA_REPO   = "diabetes-data"
BRANCH      = "main"
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"
SYD_TZ      = ZoneInfo("Australia/Sydney")

# ──────────────────────────────────────────────────────────────
# 2. Helpers
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
    return {"app_start": datetime.now(timezone.utc), "last_refresh": None, "version": "v4"}

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
# 3. Data loading
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
    return pd.DataFrame(rows).sort_values("month_date").reset_index(drop=True)

# fixed header signature
HEADER_START = "Index,Date,Time,New Device Time,BG Source,BG Reading (mmol/L)"

@st.cache_data(ttl=3600)
def load_month_csv(filename: str) -> pd.DataFrame:
    clean = str(filename).strip()
    url = RAW_BASE + quote(clean)
    if DEBUG: st.code(f"Fetching: {url}")

    # locate header line number
    with pd.read_csv(url, header=None, sep=",", engine="python", nrows=200,
                     on_bad_lines="skip", encoding="utf-8-sig") as reader:
        lines = reader
    header_row = None
    for i, row in enumerate(lines.values.tolist()):
        joined = ",".join(str(x) for x in row)
        if HEADER_START in joined:
            header_row = i
            break
    if header_row is None:
        header_row = 0  # fallback

    # load full file starting at header_row
    df = pd.read_csv(
        url, sep=",", engine="python", header=header_row,
        on_bad_lines="skip", encoding="utf-8-sig", quotechar='"'
    )
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df["month"] = extract_month_str(clean)

    # build datetime
    if "Date" in df.columns and "Time" in df.columns:
        try:
            dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                                errors="coerce", dayfirst=True)
            df["dt"] = dt.dt.tz_localize(SYD_TZ, nonexistent="NaT", ambiguous="NaT")
        except Exception:
            df["dt"] = pd.NaT

    if DEBUG:
        st.write(f"Header row detected: {header_row}")
        st.write("Columns:", list(df.columns)[:14])
        st.dataframe(df.head(10))
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
# 4. Sidebar + UI
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
# 5. Main app body
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")

render_status_sidebar()
with st.spinner("Loading index…"):
    meta = load_index()

view_mode, sel_month = month_selector(meta)
if view_mode == "all":
    data = load_all_months(meta)
elif view_mode == "single":
    fname = meta.loc[meta["month"] == sel_month, "file"].iloc[0]
    data = load_month_csv(fname)
else:
    latest = meta["month"].iloc[-1]
    fname = meta.loc[meta["month"] == latest, "file"].iloc[0]
    data = load_month_csv(fname)

# ──────────────────────────────────────────────────────────────
# 6. Summary + preview
# ──────────────────────────────────────────────────────────────
st.subheader("Summary")
st.write(f"Months: {len(meta)}, Range: {meta['month'].iloc[0]} → {meta['month'].iloc[-1]}")
st.write(f"Rows loaded: {len(data)}")

st.divider()
st.subheader("Raw preview")
st.dataframe(data.head(200), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 7. Simple chart preview (sensor glucose)
# ──────────────────────────────────────────────────────────────
GLU = "Sensor Glucose (mmol/L)"
if GLU in data.columns:
    st.line_chart(data[[ "dt", GLU ]].set_index("dt"))
else:
    st.caption("Glucose column not yet detected — verify column name.")
