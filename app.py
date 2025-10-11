# app.py — Diabetes Trends (CareLink 780G CSVs via GitHub)
import os, re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from urllib.parse import quote
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# 0. Query params
# ─────────────────────────────────────────────
qp = st.experimental_get_query_params()
if qp.get("ping") in (["1"], "1"):
    st.write("ok")
    st.stop()
DEBUG = qp.get("debug") in (["1"], "1")

# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────
GITHUB_USER = "stuarthaughey"
DATA_REPO   = "diabetes-data"
BRANCH      = "main"
RAW_BASE    = f"https://raw.githubusercontent.com/{GITHUB_USER}/{DATA_REPO}/{BRANCH}/"
SYD_TZ      = ZoneInfo("Australia/Sydney")

# ─────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────
MONTH_PATTERN = re.compile(r"([A-Za-z]{3}-\d{4})\.csv$")
def extract_month_str(fname:str)->str:
    m = MONTH_PATTERN.search(str(fname).strip())
    return m.group(1) if m else "Unknown"

def parse_month_to_date(mmm_yyyy:str)->datetime:
    return datetime.strptime(mmm_yyyy,"%b-%Y")

@st.cache_resource
def get_status():
    return {"app_start": datetime.now(timezone.utc), "last_refresh": None}

def fmt_dt(dt):
    if not dt: return "—"
    return dt.astimezone(SYD_TZ).strftime("%a %d %b %Y, %I:%M:%S %p AEST")

def human_delta(td):
    s=int(td.total_seconds())
    if s<60: return f"{s}s"
    m,s=divmod(s,60)
    if m<60: return f"{m}m {s}s"
    h,m=divmod(m,60)
    if h<24: return f"{h}h {m}m"
    d,h=divmod(h,24)
    return f"{d}d {h}h"

# ─────────────────────────────────────────────
# 3. Loaders
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_index():
    url=RAW_BASE+"index.csv"
    idx=pd.read_csv(url)
    rows=[]
    for f in idx["file"].dropna():
        m=extract_month_str(f)
        d=parse_month_to_date(m)
        rows.append({"file":f,"month":m,"month_date":d})
    return pd.DataFrame(rows).sort_values("month_date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def find_data_start(url:str)->int:
    """Finds the line index where the true data table starts (the 'Index,Date,Time' header)."""
    with open(pd.compat.get_handle(url, mode="r", encoding="utf-8-sig")[0].name) as f:
        pass  # just to hint this line; not used in cloud, fallback below

    # fallback remote read
    txt = pd.read_csv(url, header=None, sep="\n", nrows=200, encoding="utf-8-sig").astype(str)
    for i, line in enumerate(txt[0].tolist()):
        if line.startswith("Index,Date,Time"):
            return i
    return 0

@st.cache_data(ttl=3600)
def load_month_csv(filename:str)->pd.DataFrame:
    clean=str(filename).strip()
    url=RAW_BASE+quote(clean)
    if DEBUG: st.code(f"Fetching: {url}")

    # find header line
    txt=pd.read_csv(url, header=None, sep="\n", encoding="utf-8-sig", on_bad_lines="skip")
    header_row=None
    for i,line in enumerate(txt[0].astype(str).tolist()):
        if line.startswith("Index,Date,Time"):
            header_row=i
            break
    if header_row is None:
        raise ValueError("Could not find header line starting with Index,Date,Time")

    df=pd.read_csv(url, sep=",", engine="python", header=header_row,
                   on_bad_lines="skip", encoding="utf-8-sig", quotechar='"')
    df=df.dropna(axis=1,how="all")
    df.columns=[str(c).strip() for c in df.columns]
    df["month"]=extract_month_str(clean)

    if "Date" in df.columns and "Time" in df.columns:
        try:
            dt=pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str),
                              errors="coerce",dayfirst=True)
            df["dt"]=dt.dt.tz_localize(SYD_TZ,nonexistent="NaT",ambiguous="NaT")
        except Exception:
            df["dt"]=pd.NaT

    if DEBUG:
        st.write(f"Header row detected at line {header_row}")
        st.write("Columns:",list(df.columns)[:14])
        st.dataframe(df.head(10))
    return df

@st.cache_data(ttl=3600)
def load_all_months(meta):
    frames=[]
    for f in meta["file"].tolist():
        try:
            frames.append(load_month_csv(f))
        except Exception as e:
            st.warning(f"Could not load {f}: {e}")
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

# ─────────────────────────────────────────────
# 4. UI
# ─────────────────────────────────────────────
def render_sidebar():
    s=get_status()
    now=datetime.now(timezone.utc)
    st.sidebar.markdown("### App Status")
    st.sidebar.write(f"Start: {fmt_dt(s['app_start'])}")
    st.sidebar.write(f"Uptime: {human_delta(now-s['app_start'])}")
    st.sidebar.write(f"Last refresh: {fmt_dt(s['last_refresh'])}")
    st.sidebar.caption("Health ?ping=1 • Debug ?debug=1")

def month_selector(meta):
    months=meta["month"].tolist()
    latest=months[-1]
    mode=st.sidebar.radio("View",["Latest","Single","All"],index=0)
    if mode=="Latest": return "latest",latest
    elif mode=="Single":
        sel=st.sidebar.selectbox("Choose month",months,index=len(months)-1)
        return "single",sel
    else: return "all",None

# ─────────────────────────────────────────────
# 5. Main app
# ─────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Trends", layout="wide")
st.title("Diabetes Trends")
render_sidebar()

meta=load_index()
view,sel=month_selector(meta)

if view=="all":
    data=load_all_months(meta)
elif view=="single":
    fname=meta.loc[meta["month"]==sel,"file"].iloc[0]
    data=load_month_csv(fname)
else:
    latest=meta["month"].iloc[-1]
    fname=meta.loc[meta["month"]==latest,"file"].iloc[0]
    data=load_month_csv(fname)

get_status()["last_refresh"]=datetime.now(timezone.utc)

st.subheader("Summary")
st.write(f"Months: {len(meta)}, Range: {meta['month'].iloc[0]} → {meta['month'].iloc[-1]}")
st.write(f"Rows loaded: {len(data)}")

st.divider()
st.subheader("Raw preview")
st.dataframe(data.head(200),use_container_width=True)

GLU="Sensor Glucose (mmol/L)"
if GLU in data.columns and "dt" in data.columns:
    st.subheader("Sensor Glucose Over Time")
    st.line_chart(data[["dt",GLU]].set_index("dt"))
else:
    st.caption("Glucose column not found yet — verify header name and rerun.")
