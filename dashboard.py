import os
import json
import joblib
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import requests
from supabase import create_client

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
AQICN_TOKEN   = os.getenv("AQICN_TOKEN")
OW_KEY        = os.getenv("OPENWEATHER_KEY")
CITY          = os.getenv("CITY", "Islamabad")
FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
MODEL_DIR     = "models"
LAT, LON      = 33.6844, 73.0479

FEATURE_COLS = [
    "temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_direction",
    "precipitation", "weather_code",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dayofweek", "is_weekend", "is_rush_hour", "season",
    "is_hot", "is_cold", "is_calm_wind", "is_strong_wind", "is_stagnant",
    "temp_humidity", "wind_humidity",
    "aqi_lag_72h", "aqi_lag_96h",
    "aqi_rolling_72h", "aqi_rolling_96h"
]

def aqi_category(aqi):
    if aqi <= 50:   return "Good",                    "#16a34a", "😊"
    if aqi <= 100:  return "Moderate",                "#ca8a04", "😐"
    if aqi <= 150:  return "Unhealthy for Sensitive", "#ea580c", "😷"
    if aqi <= 200:  return "Unhealthy",               "#dc2626", "🤢"
    if aqi <= 300:  return "Very Unhealthy",          "#9333ea", "🚨"
    return              "Hazardous",                  "#9f1239", "☠️"

def hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color + alpha float (0-1) → 'rgba(r,g,b,a)'.
    Plotly does NOT accept 8-digit hex strings like #rrggbbaa."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first st call
# ─────────────────────────────────────────────
from PIL import Image

# icon = Image.open("c:\\Users\\Dell\\Desktop\\aqi-icon.png")
st.set_page_config(
    page_title=f"AQI Predictor — {CITY}",
    page_icon="🌬️",
    # page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# THEME — read Streamlit's actual setting
# ─────────────────────────────────────────────
# Initialize theme in session state so the toggle persists across reruns
if "theme" not in st.session_state:
    # Try to auto-detect from Streamlit config; default to dark
    try:
        base = st.get_option("theme.base")
        st.session_state.theme = "light" if base == "light" else "dark"
    except Exception:
        st.session_state.theme = "dark"

IS_DARK = (st.session_state.theme == "dark")

# ── Concrete color values for each theme (no CSS vars in SVGs!) ──
if IS_DARK:
    # BG0      = "#0b0f1a"
    # BG1      = "#131c2e"
    # BG2      = "#1a2540"
    # BORDER   = "rgba(255,255,255,0.07)"
    # BORDER2  = "rgba(255,255,255,0.14)"
    # T1       = "#f0f4ff"        # headings / primary text
    # T2       = "#8ea3c3"        # secondary text
    # T3       = "#4a6080"        # muted / labels
    # ACCENT   = "#818cf8"
    # ACCENT_BG= "#1e2047"
    # SHADOW   = "0 2px 16px rgba(0,0,0,0.5)"
    # SHADOW_LG= "0 8px 40px rgba(0,0,0,0.7)"
    # PLOT_FONT= "#c0cfe8"
    # PLOT_GRID= "rgba(255,255,255,0.06)"
    # PLOT_LINE= "rgba(255,255,255,0.1)"
    
    BG0      = "#0b0f1a"
    BG1      = "#131c2e"
    BG2      = "#1a2540"
    BORDER   = "rgba(255,255,255,0.2)"
    BORDER2  = "rgba(255,255,255,0.14)"
    T1       = "#f0f4ff"        # headings / primary text
    T2       = "#8ea3c3"        # secondary text
    T3       = "#4a6080"        # muted / labels
    ACCENT   = "#818cf8"
    ACCENT_BG= "#363a8a"
    SHADOW   = "0 2px 16px rgba(0,0,0,0.5)"
    SHADOW_LG= "0 8px 40px rgba(0,0,0,0.7)"
    PLOT_FONT= "#c0cfe8"
    PLOT_GRID= "rgba(255,255,255,0.06)"
    PLOT_LINE= "rgba(255,255,255,0.1)"
else:
    BG0      = "#eef1f7"
    BG1      = "#ffffff"
    BG2      = "#f5f7fc"
    BORDER   = "rgba(0,0,0,0.08)"
    BORDER2  = "rgba(0,0,0,0.15)"
    T1       = "#0d1117"
    T2       = "#374151"
    T3       = "#6b7280"
    ACCENT   = "#4f46e5"
    ACCENT_BG= "#eef2ff"
    SHADOW   = "0 2px 12px rgba(0,0,0,0.07)"
    SHADOW_LG= "0 8px 32px rgba(0,0,0,0.12)"
    PLOT_FONT= "#374151"
    PLOT_GRID= "rgba(0,0,0,0.06)"
    PLOT_LINE= "rgba(0,0,0,0.12)"

# ─────────────────────────────────────────────
# INJECT CSS  (uses Python vars → no CSS var issues)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem !important; padding-bottom: 3rem !important; max-width: 1380px !important; }}

/* ── App background ── */
.stApp {{ background: {BG0} !important; }}
.stApp > div {{ background: {BG0} !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background: {BG1} !important; border-right: 1px solid {BORDER2} !important; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.25rem !important; }}

/* ── Section heading ── */
.sh {{ display: flex; align-items: center; gap: 0.6rem; margin: 2.25rem 0 1rem; }}
.sh-title {{ font-size: 1.15rem; font-weight: 700; color: {T1}; letter-spacing: -0.02em; }}
.sh-pill {{
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: {ACCENT}; background: {ACCENT_BG};
    padding: 3px 10px; border-radius: 999px; border: 1px solid {ACCENT}44;
}}

/* ── Page header ── */
.ph {{ display: flex; justify-content: space-between; align-items: flex-end;
    margin-bottom: 1.5rem; padding-bottom: 1rem;
    border-bottom: 1px solid {BORDER2}; flex-wrap: wrap; gap: 0.5rem; }}
.ph-title {{ font-size: 1.8rem; font-weight: 700; color: {T1}; letter-spacing: -0.03em; }}
.ph-sub {{ font-size: 0.8rem; color: {T3}; margin-top: 3px; }}
.ph-meta {{ font-size: 0.72rem; color: {T3}; font-family: 'JetBrains Mono',monospace; text-align: right; }}

/* ── Card base ── */
.card {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 1.25rem 1.5rem;
    box-shadow: {SHADOW};
    transition: box-shadow .22s ease, transform .22s ease;
}}
.card:hover {{ box-shadow: {SHADOW_LG}; transform: translateY(-2px); }}

/* ── Gauge card ── */
.gc {{
    background: {BG1};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 1.1rem 0.75rem 0.9rem;
    box-shadow: {SHADOW};
    text-align: center;
    transition: box-shadow .22s ease, transform .22s ease;
    height: 100%;
}}
.gc:hover {{ box-shadow: {SHADOW_LG}; transform: translateY(-2px); }}

/* Gauge label BELOW svg */
.gl {{
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {T2};
    margin-top: 0.3rem;
}}
.gs {{
    font-size: 0.8rem;
    color: {T3};
    font-weight: 500;
    margin-top: 0.15rem;
}}

/* ── AQI hero ── */
.aqi-hero {{
    border-radius: 24px;
    padding: 1.75rem 1.25rem 1.5rem;
    text-align: center;
    border: 1px solid;
    box-shadow: {SHADOW_LG};
    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}}
.aqi-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 4.2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
}}
.aqi-cat {{
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}}
.live-pill {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; opacity: 0.85; margin-bottom: 0.6rem;
}}
.pulse {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor;
    animation: blink 2s ease-in-out infinite; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}

/* ── Forecast cards ── */
.fc {{
    border-radius: 18px; padding: 1.35rem;
    text-align: center; border: 1px solid;
    transition: transform .22s ease, box-shadow .22s ease;
}}
.fc:hover {{ transform: translateY(-3px); box-shadow: {SHADOW_LG}; }}
.fc-day {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.75; margin-bottom: 0.5rem; }}
.fc-aqi {{ font-family: 'JetBrains Mono',monospace; font-size: 3rem;
    font-weight: 700; letter-spacing: -0.04em; line-height: 1.05; margin: 0.3rem 0; }}
.fc-cat {{ font-size: 0.82rem; font-weight: 600; opacity: 0.9; }}
.fc-rng {{ font-family: 'JetBrains Mono',monospace; font-size: 0.7rem;
    opacity: 0.55; margin-top: 0.5rem; }}

/* ── Alert banner ── */
.alert {{
    padding: 0.9rem 1.25rem; border-radius: 14px;
    border-left: 4px solid; display: flex; align-items: center; gap: 0.75rem;
    font-size: 0.92rem; font-weight: 500; margin-bottom: 1.25rem;
    box-shadow: {SHADOW};
}}

/* ── Info box ── */
.ib {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 0.9rem 1.1rem; font-size: 0.84rem; color: {T2}; line-height: 1.75; }}
.ib strong {{ color: {T1}; }}

/* ── Perf banner ── */
.pb {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 18px;
    padding: 1.1rem 1.5rem; display: flex; gap: 2.5rem; flex-wrap: wrap; align-items: center; }}
.pb-lbl {{ font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {T3}; margin-bottom: 3px; }}
.pb-val {{ font-size: 1.25rem; font-weight: 700; font-family: 'JetBrains Mono',monospace; color: {T1}; }}

/* ── Divider ── */
.divider {{ height: 1px; background: {BORDER}; margin: 2rem 0; border: none; }}

/* ── Sidebar elements ── */
.sb-brand {{ font-size: 1.1rem; font-weight: 700; color: {T1};
    letter-spacing: -0.02em; margin-bottom: 2px; }}
.sb-city {{ font-size: 0.78rem; color: {T3}; margin-bottom: 1.25rem; }}
.sb-sect {{ font-size: 0.63rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {T3}; display: block; margin-bottom: 0.6rem; }}
.sb-row {{ display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid {BORDER}; font-size: 0.84rem; }}
.sb-key {{ color: {T2}; }}
.sb-val {{ font-family: 'JetBrains Mono',monospace; font-size: 0.8rem;
    font-weight: 600; color: {T1}; }}
.sc {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.38rem 0.7rem;
    border-radius: 999px; font-size: 0.76rem; font-weight: 600;
    margin-bottom: 5px; border: 1px solid transparent; }}
.sc-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

/* ── Streamlit tabs ── */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: {BG2};
    border-radius: 10px; padding: 4px; border: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 7px; font-size: 0.85rem;
    font-weight: 500; font-family: 'Outfit',sans-serif !important; color: {T2} !important; }}
.stTabs [aria-selected="true"] {{ background: {BG1} !important;
    box-shadow: {SHADOW}; color: {T1} !important; }}
.stTabs [data-baseweb="tab"]:hover {{ color: {T1} !important; }}

/* ── Streamlit buttons ── */
.stButton > button {{
    font-family: 'Outfit',sans-serif !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    background: {BG2} !important;
    border: 1px solid {BORDER2} !important;
    color: {T1} !important;
    transition: all .2s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: {SHADOW_LG} !important;
    background: {ACCENT} !important;
    color: #fff !important;
    border-color: {ACCENT} !important;
}}

/* Plotly chart bg */
.js-plotly-plot {{ border-radius: 16px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SVG GAUGES  — all colors hardcoded from Python vars, never CSS vars
# ─────────────────────────────────────────────

def arc_gauge(value, max_val, color, label, unit, sub,
              start_deg=215, sweep=290, size=165):
    """Arc gauge with hardcoded colors so it renders correctly in both themes."""
    pct = max(0.0, min(1.0, value / max_val))
    cx = cy = size / 2
    r  = size * 0.345
    sw = size * 0.088

    track_color = "rgba(150,150,150,0.18)" if IS_DARK else "rgba(0,0,0,0.10)"
    text_color  = T1
    unit_color  = T3

    def polar(deg, rad_r=None):
        rad = math.radians(deg)
        rad_r = rad_r or r
        return cx + rad_r * math.cos(rad), cy + rad_r * math.sin(rad)

    def arc_path(a, b):
        s, e = polar(a), polar(b)
        large = 1 if abs(b - a) > 180 else 0
        return f"M {s[0]:.2f} {s[1]:.2f} A {r} {r} 0 {large} 1 {e[0]:.2f} {e[1]:.2f}"

    val_end = start_deg + sweep * pct
    tip     = polar(val_end)

    # Subtle tick marks
    ticks = ""
    for i in range(11):
        ang  = start_deg + sweep * (i / 10)
        inn  = polar(ang, r - sw * 0.55)
        out  = polar(ang, r - sw * 0.1)
        ticks += (f'<line x1="{inn[0]:.1f}" y1="{inn[1]:.1f}" '
                  f'x2="{out[0]:.1f}" y2="{out[1]:.1f}" '
                  f'stroke="rgba(150,150,150,0.3)" stroke-width="1.2" stroke-linecap="round"/>')

    val_str = f"{value:.0f}"
    fs_val  = int(size * 0.175)
    fs_unit = int(size * 0.092)

    return f"""
<div style="text-align:center">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="overflow:visible;display:block;margin:0 auto">
    <path d="{arc_path(start_deg, start_deg+sweep)}"
          fill="none" stroke="{track_color}" stroke-width="{sw}" stroke-linecap="round"/>
    <path d="{arc_path(start_deg, val_end)}"
          fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round"
          style="filter:drop-shadow(0 0 5px {color}66)"/>
    {ticks}
    <circle cx="{tip[0]:.2f}" cy="{tip[1]:.2f}" r="{sw*0.44:.1f}" fill="{color}"
            style="filter:drop-shadow(0 0 4px {color}88)"/>
    <text x="{cx}" y="{cy+2}" text-anchor="middle" dominant-baseline="middle"
          font-family="JetBrains Mono,monospace" font-size="{fs_val}"
          font-weight="700" fill="{text_color}">{val_str}</text>
    <text x="{cx}" y="{cy + size*0.145:.0f}" text-anchor="middle"
          font-family="Outfit,sans-serif" font-size="{fs_unit}"
          font-weight="500" fill="{unit_color}">{unit}</text>
  </svg>
  <div class="gl">{label}</div>
  <div class="gs">{sub}</div>
</div>"""


def compass_gauge(wind_speed, wind_dir_deg, size=165):
    cx = cy = size / 2
    r_out = size * 0.41
    r_inn = size * 0.24
    alen  = size * 0.27

    track_color  = "rgba(150,150,150,0.18)" if IS_DARK else "rgba(0,0,0,0.10)"
    label_color  = T3
    text_color   = T1
    unit_color   = T3
    arrow_color  = ACCENT

    ang = math.radians(wind_dir_deg - 90)
    tx = cx + alen * math.cos(ang)
    ty = cy + alen * math.sin(ang)
    bx = cx - alen * 0.5 * math.cos(ang)
    by = cy - alen * 0.5 * math.sin(ang)
    px = -math.sin(ang) * size * 0.052
    py =  math.cos(ang) * size * 0.052

    labels = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
    lsvg = ""
    for lbl, deg in labels:
        rad = math.radians(deg - 90)
        lx = cx + (r_out + size * 0.062) * math.cos(rad)
        ly = cy + (r_out + size * 0.062) * math.sin(rad)
        lsvg += (f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" '
                 f'font-family="JetBrains Mono,monospace" font-size="{int(size*0.088)}" '
                 f'font-weight="600" fill="{label_color}">{lbl}</text>')

    tsvg = ""
    for i in range(36):
        rad = math.radians(i * 10 - 90)
        main = (i % 9 == 0)
        r1 = r_out - (size * 0.038 if main else size * 0.018)
        op = "0.35" if main else "0.15"
        tsvg += (f'<line x1="{cx+r1*math.cos(rad):.1f}" y1="{cy+r1*math.sin(rad):.1f}" '
                 f'x2="{cx+r_out*math.cos(rad):.1f}" y2="{cy+r_out*math.sin(rad):.1f}" '
                 f'stroke="rgba(150,150,150,{op})" stroke-width="1.2" stroke-linecap="round"/>')

    return f"""
<div style="text-align:center">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="overflow:visible;display:block;margin:0 auto">
    <circle cx="{cx}" cy="{cy}" r="{r_out}" fill="none" stroke="{track_color}" stroke-width="1.5"/>
    <circle cx="{cx}" cy="{cy}" r="{r_inn}" fill="{ACCENT}08" stroke="{ACCENT}22" stroke-width="1"/>
    {tsvg}
    {lsvg}
    <polygon points="{tx:.1f},{ty:.1f} {bx+px:.1f},{by+py:.1f} {bx-px:.1f},{by-py:.1f}"
             fill="{arrow_color}" opacity="0.95"
             style="filter:drop-shadow(0 2px 5px {arrow_color}55)"/>
    <text x="{cx}" y="{cy+2}" text-anchor="middle" dominant-baseline="middle"
          font-family="JetBrains Mono,monospace" font-size="{int(size*0.148)}"
          font-weight="700" fill="{text_color}">{wind_speed:.1f}</text>
    <text x="{cx}" y="{cy + size*0.138:.0f}" text-anchor="middle"
          font-family="Outfit,sans-serif" font-size="{int(size*0.088)}"
          font-weight="500" fill="{unit_color}">m/s</text>
  </svg>
  <div class="gl">Wind</div>
  <div class="gs">Compass direction</div>
</div>"""


def aqi_donut(aqi, size=158):
    cat, color, icon = aqi_category(aqi)
    pct = min(1.0, aqi / 500)
    cx = cy = size / 2
    r  = size * 0.365
    sw = size * 0.10

    track = "rgba(150,150,150,0.1)" if IS_DARK else "rgba(0,0,0,0.07)"

    def arc(s, e, col, w, op=1.0):
        sr = math.radians(s - 90); er = math.radians(e - 90)
        sx = cx + r * math.cos(sr); sy = cy + r * math.sin(sr)
        ex = cx + r * math.cos(er); ey = cy + r * math.sin(er)
        large = 1 if (e - s) > 180 else 0
        return (f'<path d="M {sx:.2f} {sy:.2f} A {r} {r} 0 {large} 1 {ex:.2f} {ey:.2f}" '
                f'fill="none" stroke="{col}" stroke-width="{w}" stroke-linecap="round" opacity="{op}"/>')

    vd = 360 * pct
    segs = [(0,36,"#16a34a"),(36,72,"#ca8a04"),(72,108,"#ea580c"),
            (108,144,"#dc2626"),(144,216,"#9333ea"),(216,360,"#9f1239")]
    seg_svg = "".join(arc(s, e, c, sw, 0.18) for s, e, c in segs)

    val_arc = arc(0, vd, color, sw)
    tr = math.radians(vd - 90)
    tx = cx + r * math.cos(tr); ty = cy + r * math.sin(tr)

    return f"""
<div style="text-align:center">
  <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
    {arc(0, 360, track, sw)}
    {seg_svg}
    {val_arc}
    <circle cx="{tx:.2f}" cy="{ty:.2f}" r="{sw*0.46:.1f}" fill="{color}"
            style="filter:drop-shadow(0 0 6px {color}99)"/>
    <text x="{cx}" y="{cy-8}" text-anchor="middle" dominant-baseline="middle"
          font-family="JetBrains Mono,monospace" font-size="{int(size*0.2)}"
          font-weight="700" fill="{color}">{int(aqi)}</text>
    <text x="{cx}" y="{cy+size*0.155:.0f}" text-anchor="middle"
          font-family="Outfit,sans-serif" font-size="{int(size*0.082)}"
          font-weight="600" fill="{color}" opacity="0.85">{cat}</text>
  </svg>
</div>"""


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_metadata():
    path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            content = f.read().strip()
        if not content:
            st.warning("model_metadata.json is empty — run training_pipeline.py to regenerate.")
            return {}
        return json.loads(content)
    except json.JSONDecodeError:
        st.warning("model_metadata.json is corrupted — run training_pipeline.py to regenerate.")
        return {}

@st.cache_data(ttl=3600)
def load_feature_store():
    try:
        sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        all_rows, page = [], 0
        while True:
            res = (sb.table("aqi_features").select("*").order("timestamp")
                     .range(page * 1000, (page + 1) * 1000 - 1).execute())
            if not res.data: break
            all_rows.extend(res.data)
            if len(res.data) < 1000: break
            page += 1
        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")
    except Exception as e:
        st.warning(f"Supabase: {e}")
        p = os.path.join(FEATURE_STORE, "aqi_features.csv")
        return pd.read_csv(p, parse_dates=["timestamp"]) if os.path.exists(p) else pd.DataFrame()


# Known Islamabad/Rawalpindi AQICN station feed slugs
# The bounds/map API requires a paid token — named slugs work with any free token
_ISLAMABAD_SLUGS = [
    ("US Embassy",         "islamabad"),
    ("Rawalpindi",         "rawalpindi"),
    ("Pakistan/Islamabad", "pakistan/islamabad"),
]

@st.cache_data(ttl=1800)   # recheck working stations every 30 min
def discover_islamabad_stations():
    """
    Try each known station slug and return only the ones with valid AQI.
    """
    working = []
    for name, slug in _ISLAMABAD_SLUGS:
        try:
            r = requests.get(
                f"https://api.waqi.info/feed/{slug}/?token={AQICN_TOKEN}",
                timeout=8
            ).json()
            if r.get("status") == "ok":
                aqi = r["data"]["aqi"]
                if isinstance(aqi, (int, float)) and aqi > 0:
                    working.append((name, slug))
        except Exception:
            continue
    return working if working else [("US Embassy", "islamabad")]


@st.cache_data(ttl=300)   # 5 min — so AQI feels live, not frozen for 30min
def fetch_current_aqi():
    """
    Auto-discover all Islamabad stations via AQICN bounds API,
    fetch each one, and return the city average.
    """
    stations = discover_islamabad_stations()
    readings      = []
    iaqi_combined = {}
    obs_time      = ""

    for name, key in stations:
        try:
            r = requests.get(
                f"https://api.waqi.info/feed/{key}/?token={AQICN_TOKEN}",
                timeout=8
            ).json()
            if r.get("status") == "ok":
                data = r["data"]
                aqi  = data["aqi"]
                if isinstance(aqi, (int, float)) and aqi > 0:
                    readings.append((name, int(aqi)))
                    if not iaqi_combined:   # use first station for pollutant detail
                        iaqi_combined = data.get("iaqi", {})
                        obs_time      = data.get("time", {}).get("s", "")
        except Exception:
            continue

    if not readings:
        return None, {}, "", []

    avg_aqi = round(sum(v for _, v in readings) / len(readings))
    return avg_aqi, iaqi_combined, obs_time, readings

@st.cache_data(ttl=600)   # 10 min for weather
def fetch_current_weather():
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OW_KEY}&units=metric",
            timeout=10).json()
        if r.get("cod") == 200: return r
    except: pass
    return None

def _fetch_forecast_openweather() -> pd.DataFrame:
    """Fallback forecast using OpenWeatherMap 5-day/3-hour API."""
    try:
        r = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?q={CITY}&appid={OW_KEY}&units=metric&cnt=72",
            timeout=15
        ).json()
        if r.get("cod") != "200":
            return pd.DataFrame()
        rows = []
        for item in r["list"]:
            rows.append({
                "timestamp":      pd.to_datetime(item["dt_txt"]),
                "temp":           item["main"]["temp"],
                "feels_like":     item["main"]["feels_like"],
                "humidity":       item["main"]["humidity"],
                "pressure":       item["main"]["pressure"],
                "wind_speed":     item["wind"]["speed"],
                "wind_direction": item["wind"].get("deg", 0),
                "precipitation":  item.get("rain", {}).get("3h", 0.0),
                "weather_code":   item["weather"][0]["id"],
            })
        df = pd.DataFrame(rows)
        now = pd.Timestamp.now()
        return df[df["timestamp"] > now].head(72)
    except Exception as e:
        print(f"OpenWeatherMap fallback failed: {e}")  # log silently, no UI noise
        return pd.DataFrame()


@st.cache_data(ttl=7200)  # 2hr cache — reduces Open-Meteo rate limit hits
def fetch_forecast_weather():
    try:
        headers = {"User-Agent": "PearlsAQIPredictor/1.0 (contact@example.com)"}
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LAT, "longitude": LON,
            "hourly": ["temperature_2m","relative_humidity_2m","apparent_temperature",
                       "precipitation","surface_pressure","wind_speed_10m",
                       "wind_direction_10m","weather_code"],
            "forecast_days": 4, "timezone": "Asia/Karachi"
        }, headers=headers, timeout=15)

        # 429 = rate limited, 403 = blocked — both fall back to OpenWeatherMap silently
        if resp.status_code in (429, 403):
            return _fetch_forecast_openweather()

        if resp.status_code != 200:
            st.warning(f"Forecast API returned status {resp.status_code}, trying fallback...")
            return _fetch_forecast_openweather()

        r = resp.json()

        if "error" in r or "hourly" not in r:
            return _fetch_forecast_openweather()

        df = pd.DataFrame(r["hourly"]).rename(columns={
            "time": "timestamp","temperature_2m": "temp",
            "relative_humidity_2m": "humidity","apparent_temperature": "feels_like",
            "surface_pressure": "pressure","wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
        return df[df["timestamp"] > now].head(72)
    except Exception as e:
        st.warning(f"Forecast fetch failed: {e}")
        return pd.DataFrame()

def load_shap():
    p = os.path.join(MODEL_DIR, "shap_importance.csv")
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


# ─────────────────────────────────────────────
# FORECAST ENGINE
# ─────────────────────────────────────────────
def make_forecast(model, forecast_df, last_aqi):
    if forecast_df.empty or model is None:
        return pd.DataFrame()
    hist_df = load_feature_store()
    if not hist_df.empty:
        hist_df["timestamp"] = (pd.to_datetime(hist_df["timestamp"]).dt.tz_localize(None)
                                if hist_df["timestamp"].dt.tz is None
                                else pd.to_datetime(hist_df["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None))
    rows = []
    for _, row in forecast_df.iterrows():
        ts   = row["timestamp"]
        ts_n = (ts.tz_convert("UTC").tz_localize(None)
                if hasattr(ts, "tzinfo") and ts.tzinfo else pd.Timestamp(ts))
        month, hour = ts_n.month, ts_n.hour

        def get_lag(h, _ts=ts_n):   # default arg binds ts_n NOW, not at call time
            lt = _ts - pd.Timedelta(hours=h)
            if not hist_df.empty:
                past = hist_df[hist_df["timestamp"] <= lt]
                if not past.empty: return past["aqi"].iloc[-1]
            return last_aqi

        def get_roll(h, _ts=ts_n):   # same closure fix
            et  = _ts - pd.Timedelta(hours=1)
            st2 = _ts - pd.Timedelta(hours=h)
            if not hist_df.empty:
                w = hist_df[(hist_df["timestamp"] >= st2) & (hist_df["timestamp"] <= et)]
                if not w.empty: return w["aqi"].mean()
            return last_aqi

        temp, hum, ws = row.get("temp", 30), row.get("humidity", 50), row.get("wind_speed", 3)
        feats = {
            "temp": temp, "feels_like": row.get("feels_like", temp), "humidity": hum,
            "pressure": row.get("pressure", 1010), "wind_speed": ws,
            "wind_direction": row.get("wind_direction", 180),
            "precipitation": row.get("precipitation", 0),
            "weather_code": row.get("weather_code", 0),
            "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
            "month_sin": np.sin(2*np.pi*month/12), "month_cos": np.cos(2*np.pi*month/12),
            "dayofweek": ts_n.weekday(), "is_weekend": int(ts_n.weekday() >= 5),
            "is_rush_hour": int(hour in [7,8,9,17,18,19]),
            "season": {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}.get(month, 0),
            "is_hot": int(temp>35), "is_cold": int(temp<10),
            "is_calm_wind": int(ws<2), "is_strong_wind": int(ws>8),
            "is_stagnant": int(ws<2 and hum>70),
            "temp_humidity": temp*hum, "wind_humidity": ws*hum,
            "aqi_lag_72h": get_lag(72), "aqi_lag_96h": get_lag(96),
            "aqi_rolling_72h": get_roll(72), "aqi_rolling_96h": get_roll(96),
        }
        raw_pred = float(model.predict(pd.DataFrame([feats])[FEATURE_COLS])[0])
        # Clamp prediction within ±60 of current AQI — prevents unrealistic swings
        # from synthetic training data. Remove once model is retrained on real sensor data.
        pred = max(0, min(500, round(
            np.clip(raw_pred, last_aqi * 0.45, last_aqi * 1.55), 1
        )))
        rows.append({"timestamp": ts_n, "predicted_aqi": pred, "temp": temp,
                     "humidity": hum, "wind_speed": ws,
                     "precipitation": row.get("precipitation", 0),
                     "date": ts_n.date(), "hour": hour})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PLOTLY BASE  — uses Python color vars
# ─────────────────────────────────────────────
def base_layout(**overrides):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", size=12, color=PLOT_FONT),
        margin=dict(l=0, r=10, t=40, b=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG1, font_color=T1, bordercolor=BORDER2),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor=PLOT_LINE,
                   tickfont=dict(size=11, color=PLOT_FONT)),
        yaxis=dict(gridcolor=PLOT_GRID, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(size=11, color=PLOT_FONT)),
    )
    layout.update(overrides)
    return layout


def section(title, badge=None):
    b = f'<span class="sh-pill">{badge}</span>' if badge else ""
    st.markdown(f'<div class="sh"><span class="sh-title">{title}</span>{b}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(metadata, current_aqi, aqi_obs_time="", station_readings=None):
    with st.sidebar:
        # Brand + theme toggle in same row
        c_brand, c_toggle = st.columns([3, 1])
        with c_brand:
            st.markdown(f'<div class="sb-brand">🌬️ AQI Predictor</div>'
                        f'<div class="sb-city">{CITY}, Pakistan</div>',
                        unsafe_allow_html=True)
        with c_toggle:
            st.markdown("<div style='padding-top:4px'></div>", unsafe_allow_html=True)
            moon = "☀️" if IS_DARK else "🌙"
            if st.button(moon, help="Toggle light/dark mode", key="theme_btn"):
                st.session_state.theme = "light" if IS_DARK else "dark"
                st.rerun()


        # Live AQI donut
        if current_aqi:
            cat, color, icon = aqi_category(current_aqi)
            from datetime import timezone, timedelta
            PKT = timezone(timedelta(hours=5))
            obs_label = f"<div style='font-size:0.68rem;opacity:0.55;margin-top:4px'>as of {datetime.now(PKT).strftime('%H:%M PKT, %b %d')}</div>"

            # Per-station breakdown rows
            station_rows = ""
            if station_readings:
                for sname, sval in station_readings:
                    scolor = aqi_category(sval)[1]
                    station_rows += (
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;padding:3px 0;border-bottom:1px solid {BORDER};"
                        f"font-size:0.72rem'>"
                        f"<span style='color:{T2};max-width:100px;overflow:hidden;"
                        f"text-overflow:ellipsis;white-space:nowrap'>{sname}</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;"
                        f"font-weight:700;color:{scolor}'>{sval}</span>"
                        f"</div>"
                    )
                station_rows = (
                    f"<div style='margin-top:8px;padding-top:4px'>{station_rows}"
                    f"<div style='font-size:0.67rem;color:{T3};margin-top:5px;text-align:center'>"
                    f"avg of {len(station_readings)} stations</div></div>"
                )

            st.markdown(
                f'<div style="background:{color}18;border:1px solid {color}44;'
                f'border-radius:16px;padding:1rem;margin-bottom:1.25rem;text-align:center">'
                f'<div class="live-pill" style="color:{color}">'
                f'<span class="pulse"></span>Live AQI (city avg)</div>'
                f'{aqi_donut(current_aqi, size=150)}'
                f'{obs_label}'
                f'{station_rows}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Model info
        st.markdown('<span class="sb-sect">Model Info</span>', unsafe_allow_html=True)
        for k, v in [
            ("Algorithm", metadata.get("best_model","N/A")),
            ("R² Score",  metadata.get("r2","N/A")),
            ("RMSE",      metadata.get("rmse","N/A")),
            ("MAE",       metadata.get("mae","N/A")),
            ("Trained",   str(metadata.get("trained_at","N/A"))[:10]),
        ]:
            st.markdown(
                f'<div class="sb-row">'
                f'<span class="sb-key">{k}</span>'
                f'<span class="sb-val">{v}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # AQI Scale
        st.markdown('<span class="sb-sect">AQI Scale</span>', unsafe_allow_html=True)
        for color, lbl, rng in [
            ("#16a34a","Good","0–50"),
            ("#ca8a04","Moderate","51–100"),
            ("#ea580c","Sensitive","101–150"),
            ("#dc2626","Unhealthy","151–200"),
            ("#9333ea","Very Unhealthy","201–300"),
            ("#9f1239","Hazardous","300+"),
        ]:
            st.markdown(
                f'<div class="sc" style="background:{color}18;border-color:{color}38">'
                f'<span class="sc-dot" style="background:{color}"></span>'
                f'<span style="color:{color};flex:1">{lbl}</span>'
                f'<span style="color:{color};opacity:0.65;font-family:JetBrains Mono,monospace;'
                f'font-size:0.7rem">{rng}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↻  Refresh Data", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model    = load_model()
    metadata = load_metadata()
    hist_df  = load_feature_store()
    shap_imp = load_shap()

    current_aqi, iaqi, aqi_obs_time, station_readings = fetch_current_aqi()
    weather            = fetch_current_weather()
    forecast_weather   = fetch_forecast_weather()

    render_sidebar(metadata, current_aqi, aqi_obs_time, station_readings)

    # ── PAGE HEADER ──────────────────────────
    from datetime import timezone, timedelta
    PKT = timezone(timedelta(hours=5))
    now_pkt = datetime.now(PKT)
    st.markdown(
        f'<div class="ph">'
        f'<div><div class="ph-title">Air Quality Dashboard</div>'
        f'<div class="ph-sub">{CITY}, Pakistan · 3-day ML forecast</div></div>'
        f'<div class="ph-meta">'
        f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
        f'background:#22c55e;margin-right:5px;animation:blink 2s infinite;vertical-align:middle"></span>'
        f'Updated {now_pkt.strftime("%b %d, %Y  %H:%M")} PKT · {metadata.get("best_model","XGBoost")}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # ── ALERT ────────────────────────────────
    if current_aqi and current_aqi > 150:
        cat, color, icon = aqi_category(current_aqi)
        st.markdown(
            f'<div class="alert" style="background:{color}14;border-left-color:{color};color:{color}">'
            f'<span style="font-size:1.3rem">{icon}</span>'
            f'<span>Air quality alert — AQI {current_aqi} ({cat}). '
            f'Sensitive groups should limit outdoor exposure.</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── CURRENT CONDITIONS ────────────────────
    section("Current Conditions", "Live")

    w    = weather or {}
    wm   = w.get("main", {})
    ww   = w.get("wind", {})
    temp_val  = wm.get("temp", 25)
    feels_val = wm.get("feels_like", temp_val)
    hum_val   = wm.get("humidity", 50)
    pres_val  = wm.get("pressure", 1013)
    ws_val    = ww.get("speed", 0)
    wd_val    = ww.get("deg", 0)

    if temp_val >= 40:   tc = "#ef4444"
    elif temp_val >= 30: tc = "#f97316"
    elif temp_val >= 20: tc = "#eab308"
    elif temp_val >= 10: tc = "#22c55e"
    else:                tc = "#60a5fa"

    hc = "#60a5fa" if hum_val < 40 else "#22c55e" if hum_val < 70 else "#a855f7"

    col_aqi, col_t, col_h, col_w, col_p = st.columns([1.25, 1, 1, 1, 1])

    with col_aqi:
        if current_aqi:
            _, color, icon = aqi_category(current_aqi)
            st.markdown(
                f'<div class="aqi-hero" style="background:{color}16;border-color:{color}44;color:{color}">'
                f'<div class="live-pill"><span class="pulse"></span>Current AQI</div>'
                f'<div class="aqi-num">{current_aqi}</div>'
                f'<div class="aqi-cat">{icon} {aqi_category(current_aqi)[0]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="card" style="text-align:center;padding:2rem;color:{T3}">No AQI data</div>',
                unsafe_allow_html=True
            )

    with col_t:
        st.markdown(
            f'<div class="gc">'
            + arc_gauge(temp_val, 50, tc, "Temperature", "°C",
                        f"Feels {feels_val:.1f}°C", size=160)
            + '</div>', unsafe_allow_html=True)

    with col_h:
        st.markdown(
            f'<div class="gc">'
            + arc_gauge(hum_val, 100, hc, "Humidity", "%",
                        "Relative humidity", size=160)
            + '</div>', unsafe_allow_html=True)

    with col_w:
        st.markdown(
            f'<div class="gc">'
            + compass_gauge(ws_val, wd_val, size=160)
            + '</div>', unsafe_allow_html=True)

    with col_p:
        sub_p = "Normal" if 1005 <= pres_val <= 1025 else ("High" if pres_val > 1025 else "Low")
        st.markdown(
            f'<div class="gc">'
            + arc_gauge(pres_val, 1050, ACCENT, "Pressure", "hPa",
                        sub_p, start_deg=180, sweep=180, size=160)
            + '</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 3-DAY FORECAST ────────────────────────
    section("3-Day AQI Forecast", "ML Prediction")

    if model is None:
        st.error("Model not found — run `training_pipeline.py` first.")
        return

        # Use last stored AQI from feature store if live fetch returned None
    _fallback_aqi = current_aqi
    if _fallback_aqi is None and not hist_df.empty and "aqi" in hist_df.columns:
        _fallback_aqi = float(hist_df["aqi"].iloc[-1])
    forecast_df = make_forecast(model, forecast_weather, _fallback_aqi or 100)

    if not forecast_df.empty:
        days = sorted(forecast_df["date"].unique())[:3]
        d1, d2, d3 = st.columns(3)
        for col, day in zip([d1, d2, d3], days):
            dd  = forecast_df[forecast_df["date"] == day]
            avg = dd["predicted_aqi"].mean()
            mx  = dd["predicted_aqi"].max()
            mn  = dd["predicted_aqi"].min()
            cat, color, icon = aqi_category(avg)
            label = pd.Timestamp(day).strftime("%A")
            dsub  = pd.Timestamp(day).strftime("%b %d")
            with col:
                st.markdown(
                    f'<div class="fc" style="background:{color}12;border-color:{color}38;color:{color}">'
                    f'<div class="fc-day">{label} · {dsub}</div>'
                    f'<div class="fc-aqi">{avg:.0f}</div>'
                    f'<div class="fc-cat">{icon} {cat}</div>'
                    f'<div class="fc-rng">↓ {mn:.0f}  ·  ↑ {mx:.0f}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()
        for y0, y1, rgba in [
            (0,   50,  "rgba(22,163,74,0.05)"),
            (50,  100, "rgba(202,138,4,0.06)"),
            (100, 150, "rgba(234,88,12,0.07)"),
            (150, 300, "rgba(220,38,38,0.07)"),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=rgba, line_width=0)
        for lvl, col_hex in [(50,"#16a34a"),(100,"#ca8a04"),(150,"#ea580c"),(200,"#dc2626")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=col_hex, line_width=1, opacity=0.45)
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["predicted_aqi"],
            mode="lines", line=dict(color=ACCENT, width=2.5, shape="spline", smoothing=0.8),
            fill="tozeroy", fillcolor=hex_alpha(ACCENT, 0.08),
            hovertemplate="<b>AQI %{y:.0f}</b><br>%{x|%a %b %d %H:%M}<extra></extra>"
        ))
        fig.update_layout(**base_layout(
            title=dict(text="Hourly AQI — next 72 hours",
                       font=dict(size=13, weight=600, color=T1), x=0),
            height=310, showlegend=False,
            yaxis=dict(title="AQI", gridcolor=PLOT_GRID,
                       tickfont=dict(size=11, color=PLOT_FONT),
                       range=[0, max(forecast_df["predicted_aqi"].max() * 1.2, 200)]),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── HISTORICAL TRENDS ─────────────────────
    section("Historical Trends")

    if not hist_df.empty:
        t1, t2, t3 = st.tabs(["Daily average || ", "By hour || ", "By month"])

        with t1:
            daily = hist_df.groupby(hist_df["timestamp"].dt.date)["aqi"].mean().reset_index()
            daily.columns = ["date", "avg_aqi"]
            fig1 = go.Figure(go.Scatter(
                x=daily["date"], y=daily["avg_aqi"], mode="lines",
                line=dict(color=ACCENT, width=2, shape="spline"),
                fill="tozeroy", fillcolor=hex_alpha(ACCENT, 0.07),
                hovertemplate="<b>%{x}</b><br>AQI %{y:.0f}<extra></extra>"
            ))
            fig1.update_layout(**base_layout(
                title=dict(text="Daily average AQI", font=dict(size=13, color=T1), x=0),
                height=280))
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        with t2:
            hourly = hist_df.groupby("hour")["aqi"].mean().reset_index()
            bar_c  = [aqi_category(v)[1] for v in hourly["aqi"]]
            fig2   = go.Figure(go.Bar(
                x=hourly["hour"], y=hourly["aqi"],
                marker_color=bar_c, marker_line_width=0,
                hovertemplate="<b>%{x}:00</b><br>Avg AQI %{y:.0f}<extra></extra>"
            ))
            fig2.update_layout(**base_layout(
                title=dict(text="Average AQI by hour of day", font=dict(size=13, color=T1), x=0),
                height=280, bargap=0.25))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        with t3:
            monthly = hist_df.groupby("month")["aqi"].mean().reset_index()
            mnames  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["month_name"] = monthly["month"].map(mnames)
            bar_c3 = [aqi_category(v)[1] for v in monthly["aqi"]]
            fig3   = go.Figure(go.Bar(
                x=monthly["month_name"], y=monthly["aqi"],
                marker_color=bar_c3, marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Avg AQI %{y:.0f}<extra></extra>"
            ))
            fig3.update_layout(**base_layout(
                title=dict(text="Average AQI by month", font=dict(size=13, color=T1), x=0),
                height=280, bargap=0.3))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SHAP ─────────────────────────────────
    section("What Drives AQI?", "SHAP")

    if shap_imp is not None:
        sdf = shap_imp.reset_index()
        sdf.columns = ["feature", "importance"]
        sdf = sdf.sort_values("importance", ascending=True).tail(10)
        name_map = {
            "wind_speed":"Wind speed","aqi_rolling_72h":"AQI 3-day avg",
            "aqi_lag_72h":"AQI lag 72h","aqi_lag_96h":"AQI lag 96h",
            "aqi_rolling_96h":"AQI 4-day avg","month_cos":"Month (cyclical)",
            "month_sin":"Month Sin","hour_cos":"Hour (cyclical)","hour_sin":"Hour Sin",
            "is_rush_hour":"Rush hour","humidity":"Humidity","temp":"Temperature",
            "temp_humidity":"Temp × humidity","wind_humidity":"Wind × humidity",
            "is_stagnant":"Stagnant air","precipitation":"Precipitation",
        }
        sdf["feat_clean"] = sdf["feature"].map(
            lambda x: name_map.get(x, x.replace("_", " ").title()))

        fig4 = go.Figure(go.Bar(
            x=sdf["importance"], y=sdf["feat_clean"], orientation="h",
            marker=dict(
                color=sdf["importance"],
                colorscale=[[0, hex_alpha(ACCENT, 0.25)], [1, ACCENT]],
                line_width=0,
            ),
            hovertemplate="<b>%{y}</b><br>SHAP %{x:.4f}<extra></extra>"
        ))
        fig4.update_layout(**base_layout(
            title=dict(text="Top 10 features by SHAP importance",
                       font=dict(size=13, color=T1), x=0),
            height=340,
            xaxis=dict(title="Mean |SHAP|", gridcolor=PLOT_GRID,
                       tickfont=dict(size=11, color=PLOT_FONT)),
            yaxis=dict(tickfont=dict(size=12, color=T2), linecolor="rgba(0,0,0,0)"),
        ))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        

        st.markdown(
            '<div class="ib"><strong>How to read this:</strong><br>'
            '<strong>longer bar</strong> = stronger influence on predictions.<br>'
            '<strong>Wind speed</strong> disperses pollution.<br>'
            '<strong>Rush hour</strong> spikes emissions.<br>'
            '<strong>Humidity</strong> traps particles.<br>'
            '<strong>Month</strong> captures seasonal smog cycles.</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── MODEL COMPARISON ──────────────────────
    section("Model Comparison")

    if metadata.get("all_results"):
        rdf  = pd.DataFrame(metadata["all_results"])
        best = metadata.get("best_model", "N/A")

        st.markdown(
            f'<div class="pb">'
            f'<div><div class="pb-lbl">Best Model</div>'
            f'<div class="pb-val">{best}</div></div>'
            f'<div><div class="pb-lbl">R² Score</div>'
            f'<div class="pb-val" style="color:#22c55e">{metadata.get("r2","—")}</div></div>'
            f'<div><div class="pb-lbl">RMSE</div>'
            f'<div class="pb-val">{metadata.get("rmse","—")}</div></div>'
            f'<div><div class="pb-lbl">MAE</div>'
            f'<div class="pb-val">{metadata.get("mae","—")}</div></div>'
            f'<div><div class="pb-lbl">Trained</div>'
            f'<div class="pb-val" style="font-size:0.9rem">'
            f'{str(metadata.get("trained_at","—"))[:10]}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        bc = [ACCENT if r["model"] == best else hex_alpha(ACCENT, 0.28) for _, r in rdf.iterrows()]
        gc = ["#22c55e" if r["model"] == best else hex_alpha("#22c55e", 0.28) for _, r in rdf.iterrows()]

        with c1:
            f5 = go.Figure(go.Bar(
                x=rdf["model"], y=rdf["rmse"],
                marker_color=bc, marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>RMSE %{y:.2f}<extra></extra>"
            ))
            f5.update_layout(**base_layout(
                title=dict(text="RMSE — lower is better",
                           font=dict(size=13, color=T1), x=0),
                height=260, bargap=0.38,
                xaxis=dict(tickangle=-15, tickfont=dict(size=10, color=PLOT_FONT),
                           showgrid=False, linecolor=PLOT_LINE),
            ))
            st.plotly_chart(f5, use_container_width=True, config={"displayModeBar": False})

        with c2:
            f6 = go.Figure(go.Bar(
                x=rdf["model"], y=rdf["r2"],
                marker_color=gc, marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>R² %{y:.4f}<extra></extra>"
            ))
            f6.update_layout(**base_layout(
                title=dict(text="R² score — higher is better",
                           font=dict(size=13, color=T1), x=0),
                height=260, bargap=0.38,
                xaxis=dict(tickangle=-15, tickfont=dict(size=10, color=PLOT_FONT),
                           showgrid=False, linecolor=PLOT_LINE),
            ))
            st.plotly_chart(f6, use_container_width=True, config={"displayModeBar": False})

    # ── FOOTER ───────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-size:0.74rem;color:{T3};padding-bottom:1rem">'
        f'Pearls AQI Predictor · {CITY} · '
        f'Data: AQICN · OpenWeatherMap · Open-Meteo · '
        f'Automated via GitHub Actions'
        f'</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()