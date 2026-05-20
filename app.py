"""
AI Pricing Intelligence Platform — Professional Enterprise Dashboard

Complete Streamlit enterprise application that integrates data intake,
batch pricing analysis, forecasting, competitive intelligence, inventory
optimization, risk scoring, explainability, dashboards, reporting, and chat.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.ai_assistant import AIAssistant
import dashboard.visualizations as viz
from modules.alerts import AlertEngine
from modules.competitor_engine import CompetitorEngine
from modules.csv_processor import CSVProcessor, DataValidationError
from modules.demand_forecaster import DemandForecaster
from modules.explainability import ExplainabilityEngine
from modules.inventory_engine import InventoryEngine
from modules.pricing_engine import PricingEngine
from modules.risk_engine import RiskEngine
from utils.config import AppConfig
from utils.helpers import format_currency
from utils.logger import setup_logging

try:
    from modules.reporting import ReportGenerator
    REPORTING_AVAILABLE = True
    REPORTING_IMPORT_ERROR = ""
except Exception as exc:
    ReportGenerator = None
    REPORTING_AVAILABLE = False
    REPORTING_IMPORT_ERROR = str(exc)


st.set_page_config(
    page_title="AI Pricing Intelligence Platform",
    page_icon="\U0001f916",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logging(log_level="INFO", logger_name="ai_pricing_streamlit")
SAMPLE_DATA_PATH = ROOT / "data" / "sample_products.csv"

NAV_ITEMS = [
    "Executive Overview", "Data Intake", "Batch Analysis",
    "Pricing Engine", "Demand Forecasting", "Competitor Intelligence",
    "Inventory Optimization", "Risk & Explainability",
    "Reports", "Chat Assistant",
]


# ═══════════════════════════════════════════════════════════════════════════
# ENTERPRISE CSS  —  Professional dashboard styling with Light Blue Grey palette
# ═══════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown("""
    <style>
        :root {
            --primary: #2563EB;
            --primary-dark: #1D4ED8;
            --indigo: #4F46E5;
            --emerald: #10B981;
            --danger: #EF4444;
            --slate-950: #0F172A;
            --slate-800: #1E293B;
            --slate-600: #475569;
            --slate-500: #64748B;
            --slate-200: #E2E8F0;
            --slate-100: #F1F5F9;
            --slate-50: #F8FAFC;
            --surface: #FFFFFF;
            --border: #E5E7EB;
            --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
            --shadow-md: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        .stApp {
            background-color: #F6F8FC;
            color: var(--slate-950);
        }

        header[data-testid="stHeader"] {
            background: rgba(248, 250, 252, 0.88);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226, 232, 240, 0.8);
        }

        .block-container {
            padding: 1.5rem 1.65rem 1.25rem !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        .main .block-container { overflow: visible !important; }

        #MainMenu, footer { visibility: hidden; display: none; }

        /* Force h1 visibility — fix clipping from Streamlit internal layouts */
        h1 {
            position: relative !important;
            top: 0 !important;
            margin-top: 0 !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        div[data-testid="stAppViewContainer"] {
            overflow: visible !important;
        }

        /* Page header container — ensures title is fully visible with proper spacing */
        .page-header-container {
            margin-top: 10px;
            margin-bottom: 6px;
            overflow: visible;
        }
        .page-header-container h1 {
            margin: 0 0 2px 0;
            padding: 0;
        }

        /* Typography */
        .app-title {
            font-size: 2rem; font-weight: 800; color: var(--slate-950);
            letter-spacing: 0; line-height: 1.12; margin: 0 0 0.15rem;
            padding: 0; overflow: visible; white-space: normal;
        }
        .app-sub {
            font-size: 0.9rem; color: var(--slate-500);
            margin: 0 0 1rem; font-weight: 400; line-height: 1.45;
        }
        .sec-title {
            font-size: 1.34rem; font-weight: 750; color: var(--slate-950);
            margin: 0 0 0.1rem; padding: 0; letter-spacing: 0;
            overflow: visible; white-space: normal; display: block;
            line-height: 1.25;
        }
        .sec-desc { color: var(--slate-500); font-size: 0.86rem; margin: 0 0 0.75rem; }
        .sub-title {
            font-size: 1rem; font-weight: 700; color: var(--slate-800);
            margin: 0.55rem 0 0.45rem;
        }

        /* Cards and containers - professional rounded cards with soft shadows */
        div[data-testid="stVerticalBlock"] > div.stCard {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }
        div[data-testid="stVerticalBlock"] > div.stCard:hover {
            border-color: #CBD5E1;
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        div[data-testid="stVerticalBlock"] > div.stCardChart {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem 0.5rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: border-color 0.18s ease, box-shadow 0.18s ease;
        }

        /* Metrics — professional rounded cards with white bg and soft shadow */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
            height: 100%;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-color: #BFDBFE;
            transform: translateY(-1px);
        }
        div[data-testid="stMetric"] label {
            color: var(--slate-500); font-size: 0.69rem; font-weight: 800;
            text-transform: uppercase; letter-spacing: 0.06em;
        }
        div[data-testid="stMetric"] > div:first-child {
            font-weight: 800; color: var(--slate-950); font-size: 1.38rem !important;
        }
        div[data-testid="stMetric"] svg { display: none; }

        /* Buttons */
        .stButton button, .stDownloadButton button {
            border-radius: 9px !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            padding: 0.42rem 0.85rem !important;
            border: 1px solid #CBD5E1 !important;
            color: var(--slate-800) !important;
            background: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            transition: background 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            border-color: #93C5FD !important;
            background: #F8FAFC !important;
            color: var(--primary-dark) !important;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.10);
            transform: translateY(-1px);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, var(--primary), var(--indigo)) !important;
            border: 1px solid rgba(37, 99, 235, 0.35) !important;
            color: #FFFFFF !important;
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
        }
        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, var(--primary-dark), #4338CA) !important;
            color: #FFFFFF !important;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
        }

        /* Sidebar — Light Blue Grey background */
        section[data-testid="stSidebar"] {
            width: 248px !important;
            min-width: 248px !important;
            background-color: #FAFBFD;
            border-right: 1px solid #E5E7EB;
            box-shadow: 8px 0 28px rgba(15, 23, 42, 0.035);
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 248px !important;
            padding: 0.9rem 0.85rem 0.85rem !important;
        }
        .css-1d391kg, .css-184tjsw, .eczjsme4 { width: 248px !important; min-width: 248px !important; }
        section[data-testid="stSidebar"] hr {
            margin: 0.65rem 0 !important;
            background: var(--slate-200);
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--slate-500);
            font-size: 0.72rem;
        }
        .sidebar-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 0.15rem 0 0.2rem; margin-bottom: 0.1rem;
        }
        .sidebar-brand span {
            display: inline-flex; align-items: center; justify-content: center;
            width: 30px; height: 30px; border-radius: 10px;
            background: linear-gradient(135deg, var(--primary), var(--indigo));
            color: #FFFFFF; font-size: 1.05rem;
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.20);
        }
        .sidebar-brand h2 {
            font-size: 1rem; font-weight: 800; color: var(--slate-950);
            margin: 0; line-height: 1.1; letter-spacing: 0;
        }
        .status-pill {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 4px 11px; border-radius: 999px;
            background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0;
            font-size: 0.73rem; font-weight: 600;
        }
        .status-pill .dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--emerald); box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.14);
        }
        .small-muted { color: #94A3B8; font-size: 0.72rem; }
        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
            min-height: 2.05rem;
            font-size: 0.78rem !important;
            padding: 0.35rem 0.65rem !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 0.65rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
        .nav-section-title {
            color: var(--slate-600) !important;
            font-size: 0.68rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0 0 0.35rem 0.15rem;
        }
        section[data-testid="stSidebar"] .stButton button {
            text-align: left;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px; background: var(--slate-100);
            border: 1px solid var(--border);
            border-radius: 10px; padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px; font-weight: 700; font-size: 0.78rem;
            padding: 0.38rem 0.78rem; color: var(--slate-600);
        }
        .stTabs [aria-selected="true"] {
            background: #FFFFFF;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            color: var(--primary-dark);
        }

        /* Data, alerts, inputs — rounded with soft shadows */
        .stAlert {
            border-radius: 12px;
            border-left-width: 4px;
            font-size: 0.84rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #E5E7EB;
            font-size: 0.78rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .stDataFrame thead tr th {
            background: var(--slate-50) !important;
            font-weight: 800;
            font-size: 0.72rem;
            color: var(--slate-600);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .streamlit-expanderHeader {
            font-weight: 700;
            font-size: 0.84rem;
            color: var(--slate-800);
            border-radius: 9px;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea {
            border-radius: 9px !important;
            border-color: #CBD5E1 !important;
        }
        div[data-baseweb="select"]:focus-within > div,
        div[data-baseweb="input"]:focus-within > div,
        textarea:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }
        hr { margin: 0.65rem 0; border: 0; height: 1px; background: var(--slate-200); }

        /* Layout safety */
        .main > div { overflow: visible !important; }
        .element-container { overflow: visible !important; }
        .row-widget { overflow: visible !important; }
        .stMarkdown { overflow: visible !important; }
        .stPlotlyChart { overflow: visible !important; }

        /* Chat formatting */
        .chat-section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--slate-950);
            margin: 0.5rem 0 0.25rem;
            padding: 0;
        }
        .chat-product-item {
            padding: 0.35rem 0;
            border-bottom: 1px solid var(--slate-100);
        }
        .chat-product-item:last-child {
            border-bottom: none;
        }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "config": AppConfig(), "processor_report": None,
        "raw_df": None, "processed_df": None, "analyzed_df": None,
        "analysis_reports": {}, "insights": {}, "alerts": [],
        "chat_history": [], "assistant": AIAssistant(),
        "nav": NAV_ITEMS[0], "last_run_at": None,
        "uploaded_name": None, "uploaded_bytes_size": None,
        "analysis_error": None, "max_workers": 2, "preview_rows": 50,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_sample_dataset(mtime: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = CSVProcessor(AppConfig())
    df = p.load_file(SAMPLE_DATA_PATH)
    df = p.engineer_features()
    return df, p.get_processing_report() or {}

@st.cache_data(show_spinner=False)
def process_uploaded_dataset(b: bytes, name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suffix = Path(name).suffix.lower()
    raw = pd.read_excel(io.BytesIO(b), engine="openpyxl") if suffix in {".xlsx", ".xls"} else pd.read_csv(io.BytesIO(b))
    p = CSVProcessor(AppConfig())
    df = p.load_dataframe(raw)
    df = p.engineer_features()
    return df, p.get_processing_report() or {}

@st.cache_data(show_spinner=False)
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_df(prefer_analyzed: bool = True) -> Optional[pd.DataFrame]:
    if prefer_analyzed and st.session_state.analyzed_df is not None:
        return st.session_state.analyzed_df
    if st.session_state.processed_df is not None:
        return st.session_state.processed_df
    return st.session_state.raw_df

def set_dataset(df: pd.DataFrame, rpt: dict, src: str) -> None:
    st.session_state.raw_df = st.session_state.processed_df = df.copy()
    st.session_state.analyzed_df = None
    st.session_state.processor_report = rpt
    st.session_state.analysis_reports = {}
    st.session_state.insights = build_insights(df)
    st.session_state.alerts = []
    st.session_state.uploaded_name = src
    st.session_state.chat_history = []
    st.session_state.assistant.clear_history()

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "competitor_price" not in out.columns and "current_price" in out.columns:
        out["competitor_price"] = out["current_price"] * 1.05
    for c, d in [("sales_volume", 100), ("inventory_level", 500), ("demand_trend", 0.5), ("price_elasticity", -1.5)]:
        if c not in out.columns: out[c] = d
    if "category" not in out.columns: out["category"] = "Uncategorized"
    if "product_name" not in out.columns: out["product_name"] = out.get("product_id", pd.Series(range(len(out)))).astype(str)
    if "profit_margin" not in out.columns and {"current_price", "cost_price"}.issubset(out.columns):
        out["profit_margin"] = ((out["current_price"] - out["cost_price"]) / out["current_price"]).clip(lower=0)
    if "revenue" not in out.columns and {"current_price", "sales_volume"}.issubset(out.columns):
        out["revenue"] = out["current_price"] * out["sales_volume"]
    if "price_vs_competitor" not in out.columns and {"current_price", "competitor_price"}.issubset(out.columns):
        out["price_vs_competitor"] = (out["current_price"] / out["competitor_price"].replace(0, pd.NA)).fillna(1.0)
    return out

def numcol(df: pd.DataFrame, c: str, d: float = 0.0) -> pd.Series:
    if c not in df.columns: return pd.Series([d] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce").fillna(d)

def build_insights(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty: return {}
    rev = numcol(df, "revenue")
    cp = numcol(df, "current_price")
    er = numcol(df, "expected_revenue")
    pm = numcol(df, "profit_margin")
    ep = numcol(df, "expected_profit")
    sv = numcol(df, "stock_value")
    rs = numcol(df, "composite_risk_score")
    i: Dict[str, Any] = {
        "total_products": len(df),
        "total_categories": int(df["category"].nunique()) if "category" in df.columns else 0,
        "avg_price": float(cp.mean()) if len(df) else 0,
        "avg_margin": float(pm.mean()) if "profit_margin" in df.columns else 0,
        "current_revenue": float(rev.sum()),
        "expected_revenue": float(er.sum()) if "expected_revenue" in df.columns else float(rev.sum()),
        "expected_profit": float(ep.sum()) if "expected_profit" in df.columns else 0,
        "total_inventory": float(numcol(df, "inventory_level").sum()),
        "stock_value": float(sv.sum()) if "stock_value" in df.columns else 0,
        "avg_risk_score": float(rs.mean()) if "composite_risk_score" in df.columns else 0,
        "high_risk_count": int(df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if "risk_level" in df.columns else 0,
    }
    for col, key in [("recommendation", "recommendation_distribution"), ("risk_level", "risk_distribution"),
                     ("stock_status", "stock_status_distribution")]:
        if col in df.columns: i[key] = df[col].value_counts().to_dict()
    return i

def r2d(r: Any) -> Dict[str, Any]:
    if r is None: return {}
    if hasattr(r, "to_dict"): return r.to_dict()
    return dict(r) if isinstance(r, dict) else {"summary": str(r)}


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    pr = st.progress(0, text="Preparing products")
    rpts: Dict[str, Any] = {}
    w = ensure_cols(df)
    try:
        with st.spinner("Forecasting demand..."):
            fd = DemandForecaster(max_workers=st.session_state.max_workers)
            w, rep = fd.batch_forecast(w); rpts["demand"] = r2d(rep); pr.progress(16, text="Demand done")
        with st.spinner("Competitor analysis..."):
            ce = CompetitorEngine(st.session_state.config)
            w, rep = ce.analyze(w); rpts["competition"] = r2d(rep); pr.progress(32, text="Competitor done")
        with st.spinner("Inventory optimization..."):
            ie = InventoryEngine(st.session_state.config)
            w, rep = ie.analyze(w); rpts["inventory"] = r2d(rep); pr.progress(48, text="Inventory done")
        with st.spinner("Risk scoring..."):
            re = RiskEngine(st.session_state.config)
            w, rep = re.assess_risk(w); rpts["risk"] = r2d(rep); pr.progress(64, text="Risk done")
        with st.spinner("Pricing optimization..."):
            pe = PricingEngine(st.session_state.config)
            w, rep = pe.optimize_all(w); rpts["pricing"] = r2d(rep); pr.progress(78, text="Pricing done")
        with st.spinner("Explainability..."):
            ee = ExplainabilityEngine(st.session_state.config)
            w, rep = ee.batch_explain(w); rpts["explainability"] = r2d(rep); pr.progress(90, text="Explainability done")
        with st.spinner("Alerts & context..."):
            ae = AlertEngine(st.session_state.config)
            alerts = ae.generate_alerts(w)
            rpts["alerts"] = ae.get_alert_summary()
            insights = build_insights(w)
            st.session_state.assistant.load_context(w, insights, alerts)
        pr.progress(100, text="Complete")
    except Exception:
        logger.exception("Batch failed"); pr.empty(); raise
    return w, rpts, alerts, insights


# ═══════════════════════════════════════════════════════════════════════════
# UI HELPERS  —  SAFE rendering wrappers to prevent React errors
# ═══════════════════════════════════════════════════════════════════════════

def page_hdr(t: str, c: str = "") -> None:
    """Render a page title and optional subtitle with safe spacing."""
    st.markdown(f"""
    <div class='page-header-container'>
        <h1 class='sec-title'>{t}</h1>
        {f"<p class='sec-desc'>{c}</p>" if c else ""}
    </div>
    """, unsafe_allow_html=True)

def fmt_money(v: float) -> str:
    try: return format_currency(float(v), 0)
    except: return "$0"

def safe_chart(fn, *a, **kw) -> Optional[go.Figure]:
    """Return a Plotly figure, or None if it fails — caller handles None."""
    try:
        fig = fn(*a, **kw)
        if fig is not None and hasattr(fig, 'data') and len(fig.data) > 0:
            # Check that data actually has values
            has_valid = False
            for t in fig.data:
                if hasattr(t, 'x') and t.x is not None and len(t.x) > 0:
                    has_valid = True
                    break
                if hasattr(t, 'y') and t.y is not None and len(t.y) > 0:
                    has_valid = True
                    break
            if has_valid:
                return fig
        return None
    except Exception:
        return None

def show_table(df_slice: pd.DataFrame) -> None:
    """Render a clean table — safe with empty/None guards."""
    if df_slice is not None and isinstance(df_slice, pd.DataFrame) and len(df_slice) > 0:
        st.dataframe(df_slice, use_container_width=True, hide_index=True)

def safe_metric(col, label: str, value: str, delta=None) -> None:
    """Render a metric safely inside a column."""
    if col is None:
        return
    try:
        col.metric(label, value, delta=delta)
    except Exception:
        try:
            col.metric(label, value)
        except Exception:
            pass

def card_container(key_prefix: str = "card") -> Any:
    """Create a styled card container. Returns the container for use with `with`."""
    return st.container()

def render_card(title: str = "", desc: str = "") -> Any:
    """Open a card container. Use with `with render_card(...):` block."""
    c = st.container()
    with c:
        if title:
            st.markdown(f"<div class='sub-title' style='margin-top:0;'>{title}</div>", unsafe_allow_html=True)
        if desc:
            st.markdown(f"<small style='color:#64748B;'>{desc}</small>", unsafe_allow_html=True)
    return c


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    st.sidebar.markdown('<div class="sidebar-brand"><span>\U0001f916</span><h2>Pricing AI</h2></div>', unsafe_allow_html=True)
    st.sidebar.caption("Enterprise Edition")

    up = st.sidebar.file_uploader("Upload catalog", type=["csv","xlsx","xls"],
        help="Required: product_id, current_price, cost_price")
    if up is not None and (st.session_state.raw_df is None or st.session_state.uploaded_name != up.name or st.session_state.uploaded_bytes_size != getattr(up,"size",None)):
        try:
            with st.sidebar.status("Processing...", expanded=False):
                df, rpt = process_uploaded_dataset(up.getvalue(), up.name)
                set_dataset(df, rpt, up.name)
                st.session_state.uploaded_bytes_size = getattr(up,"size",None)
                st.session_state.analysis_error = None
            st.sidebar.success(f"Loaded {len(df):,} products")
        except DataValidationError as e: st.sidebar.error(f"Validation: {e}")
        except Exception as e: st.sidebar.error(f"Load failed: {e}"); logger.exception("Upload")

    if st.sidebar.button("Load Sample", use_container_width=True):
        try:
            with st.sidebar.status("Loading...", expanded=False):
                df, rpt = load_sample_dataset(SAMPLE_DATA_PATH.stat().st_mtime)
                set_dataset(df, rpt, "sample_products.csv")
            st.sidebar.success(f"Loaded {len(df):,} sample products")
        except Exception as e: st.sidebar.error(f"Sample load: {e}")

    st.sidebar.divider()
    adf = get_df(prefer_analyzed=False)
    if adf is not None:
        st.sidebar.markdown(f"<div class='status-pill'><span class='dot'></span> {len(adf):,} loaded</div>", unsafe_allow_html=True)
        if st.session_state.analyzed_df is not None:
            st.sidebar.caption(f"Last run: {st.session_state.last_run_at}")
        if st.sidebar.button("Run Full Analysis", key="run_btn", type="primary", use_container_width=True):
            try:
                a, rpts, alerts, ins = run_batch(adf)
                st.session_state.analyzed_df = a; st.session_state.analysis_reports = rpts
                st.session_state.alerts = alerts; st.session_state.insights = ins
                st.session_state.last_run_at = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.analysis_error = None
                st.sidebar.success("Analysis complete.")
            except Exception as e:
                st.session_state.analysis_error = str(e); st.sidebar.error(f"Failed: {e}"); logger.exception("Batch")

    st.sidebar.divider()
    if st.session_state.nav not in NAV_ITEMS:
        st.session_state.nav = NAV_ITEMS[0]
    st.sidebar.markdown("<div class='nav-section-title'>Navigation</div>", unsafe_allow_html=True)
    for i, item in enumerate(NAV_ITEMS):
        is_active = st.session_state.nav == item
        if st.sidebar.button(
            item,
            key=f"nav_{i}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav = item
            st.rerun()

    if st.sidebar.button("Reset", use_container_width=True):
        for k in ["raw_df","processed_df","analyzed_df","processor_report","analysis_reports","insights","alerts","chat_history","uploaded_name","last_run_at"]:
            st.session_state[k] = [] if k in {"alerts","chat_history"} else ({} if k in {"analysis_reports","insights"} else None)
        st.session_state.assistant.clear_history(); st.rerun()
    st.sidebar.divider()
    st.sidebar.markdown("<div class='small-muted'>v2.0 · AI Pricing Platform</div>", unsafe_allow_html=True)


def require_data(prefer_analyzed: bool = False) -> Optional[pd.DataFrame]:
    df = get_df(prefer_analyzed=prefer_analyzed)
    if df is None: st.info("Upload a CSV/Excel or load the sample dataset from the sidebar.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TABS  —  All use safe rendering patterns, no raw HTML div wrapping Streamlit elements
# ═══════════════════════════════════════════════════════════════════════════

def render_executive_overview() -> None:
    page_hdr("Executive Overview", "Portfolio-level pricing, risk, demand, and margin signals.")
    df = require_data(prefer_analyzed=True)
    if df is None: return
    ins = st.session_state.insights or build_insights(df)
    rd = ins.get("expected_revenue", 0) - ins.get("current_revenue", 0)

    # SAFE: Use st.columns directly, not wrapped in raw HTML divs
    cols = st.columns(5)
    safe_metric(cols[0], "Products", f"{ins.get('total_products',0):,}")
    safe_metric(cols[1], "Revenue", fmt_money(ins.get("current_revenue",0)))
    safe_metric(cols[2], "Expected", fmt_money(ins.get("expected_revenue",0)), fmt_money(rd))
    safe_metric(cols[3], "Margin", f"{ins.get('avg_margin',0)*100:.1f}%")
    safe_metric(cols[4], "High Risk", f"{ins.get('high_risk_count',0):,}")

    l, r = st.columns([1.25, 1])
    with l:
        fig = safe_chart(viz.create_revenue_by_category_chart, df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Revenue chart not available yet. Run analysis first.")
    with r:
        fig = safe_chart(viz.create_recommendation_pie, df) if "recommendation" in df.columns else safe_chart(viz.create_margin_distribution_chart, df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Distribution chart not available yet. Run analysis first.")

    st.markdown("<div class='sub-title'>Priority Actions</div>", unsafe_allow_html=True)
    pcols = ["product_id","product_name","category","current_price","optimal_price","price_change_pct","recommendation","composite_risk_score","risk_level"]
    av = [c for c in pcols if c in df.columns]
    if av:
        sc = "composite_risk_score" if "composite_risk_score" in df.columns else av[0]
        sorted_df = df.sort_values(sc, ascending=False)[av].head(15)
        show_table(sorted_df)


def render_data_intake() -> None:
    page_hdr("Data Intake", "Upload, validate, and preview catalog data.")
    df = require_data()
    if df is None: return
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Source:** {st.session_state.uploaded_name or 'Loaded'}")
        st.markdown(f"**Rows:** {len(df):,}  **Cols:** {len(df.columns):,}")
        if st.session_state.processor_report:
            with st.expander("Processing Report"):
                st.json(st.session_state.processor_report)
    with c2:
        schema = pd.DataFrame({"col": df.columns, "type": [str(df[c].dtype) for c in df.columns]})
        show_table(schema)
    st.markdown("<div class='sub-title'>Data Preview</div>", unsafe_allow_html=True)
    show_table(df.head(st.session_state.preview_rows))


def render_batch_analysis() -> None:
    page_hdr("Batch Analysis", "Pipeline results across all engines.")
    df = require_data()
    if df is None: return
    if st.session_state.analysis_error: st.error(st.session_state.analysis_error)
    adf = st.session_state.analyzed_df
    if adf is None:
        st.info("Run full analysis from the sidebar to generate the consolidated output.")
        return

    st.markdown("**Analysis complete.** The consolidated analyzed dataset is available below.")
    st.markdown("<div class='sub-title'>Analyzed Output</div>", unsafe_allow_html=True)
    show_table(adf.head(st.session_state.preview_rows))
    st.download_button("Download CSV", data=df_to_csv(adf), file_name="analyzed_output.csv", mime="text/csv", use_container_width=True)


def render_pricing_engine() -> None:
    page_hdr("Pricing Engine", "Optimal price recommendations and financial impact.")
    df = require_data(prefer_analyzed=True)
    if df is None or "optimal_price" not in df.columns: st.warning("Run full analysis first."); return
    cr = numcol(df, "revenue").sum(); er = numcol(df, "expected_revenue").sum()
    cpt = ((numcol(df, "current_price") - numcol(df, "cost_price")) * numcol(df, "sales_volume")).sum()
    ep = numcol(df, "expected_profit").sum()
    cols = st.columns(4)
    safe_metric(cols[0], "Revenue Lift", fmt_money(er - cr))
    safe_metric(cols[1], "Expected Revenue", fmt_money(er))
    safe_metric(cols[2], "Profit Lift", fmt_money(ep - cpt))
    safe_metric(cols[3], "Avg Change", f"{numcol(df,'price_change_pct').mean():+.1f}%")
    l, r = st.columns(2)
    for col, fn in [(l, viz.create_price_vs_optimal_scatter), (r, viz.create_recommendation_pie)]:
        with col:
            try:
                fig = safe_chart(fn, df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No data available for chart.")
            except Exception:
                st.caption("Chart unavailable.")
    cols = ["product_name","category","current_price","optimal_price","price_change_pct","recommendation","expected_revenue","expected_profit","explanation_summary"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        sorted_df = df[available_cols].sort_values("price_change_pct", ascending=False)
        show_table(sorted_df)


def render_demand_forecasting() -> None:
    page_hdr("Demand Forecasting", "30-day projections and trend segmentation.")
    df = require_data(prefer_analyzed=True)
    if df is None or "predicted_demand" not in df.columns: st.warning("Run full analysis first."); return
    cols = st.columns(4)
    safe_metric(cols[0], "Avg Demand", f"{numcol(df,'predicted_demand').mean():,.0f}")
    safe_metric(cols[1], "Confidence", f"{numcol(df,'forecast_confidence').mean()*100:.0f}%")
    safe_metric(cols[2], "Total Forecast", f"{numcol(df,'forecast_next_30d').sum():,.0f}")
    inc = int((df.get("demand_trend_category", pd.Series(dtype=str)) == "increasing").sum())
    safe_metric(cols[3], "Increasing", f"{inc:,}")
    l, r = st.columns(2)
    with l:
        fcols = [c for c in df.columns if c.startswith("forecast_week_")]
        if fcols and len(fcols) > 0:
            wk = df[fcols].sum().reset_index()
            wk.columns = ["week","units"]
            try:
                fig = px.line(wk, x="week", y="units", markers=True, title="Weekly Forecast").update_layout(
                    margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("Forecast chart unavailable.")
        else:
            st.info("No forecast data yet.")
    with r:
        if "demand_trend_category" in df.columns:
            tc = df["demand_trend_category"].value_counts().reset_index()
            tc.columns = ["trend","count"]
            try:
                fig = px.pie(tc, values="count", names="trend", title="Trend Distribution", hole=0.4,
                    color="trend", color_discrete_map={"increasing":"#2e7d32","stable":"#1565c0","declining":"#c62828"})\
                    .update_layout(margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("Trend chart unavailable.")
    cols = ["product_name","category","sales_volume","predicted_demand","forecast_confidence","forecast_ci_lower","forecast_ci_upper","demand_trend_category"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(df[available_cols].sort_values("predicted_demand", ascending=False))


def render_competitor_intelligence() -> None:
    page_hdr("Competitor Intelligence", "Competitive pricing and threat signals.")
    df = require_data(prefer_analyzed=True)
    if df is None or "competitive_score" not in df.columns: st.warning("Run full analysis first."); return
    cols = st.columns(4)
    safe_metric(cols[0], "Avg Score", f"{numcol(df,'competitive_score').mean():.1f}/100")
    safe_metric(cols[1], "Overpriced", f"{int(df.get('overpriced_flag',pd.Series(dtype=bool)).sum()):,}")
    safe_metric(cols[2], "Underpriced", f"{int(df.get('underpriced_flag',pd.Series(dtype=bool)).sum()):,}")
    safe_metric(cols[3], "Undercuts", f"{int(df.get('competitor_undercut_flag',pd.Series(dtype=bool)).sum()):,}")
    fig = safe_chart(viz.create_competitor_price_comparison, df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    cols = ["product_name","category","current_price","competitor_price","market_position","price_gap_pct","competitive_score","pricing_recommendation"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(df[available_cols].sort_values("competitive_score"))


def render_inventory_optimization() -> None:
    page_hdr("Inventory Optimization", "Stock health and pricing actions.")
    df = require_data(prefer_analyzed=True)
    if df is None or "inventory_score" not in df.columns: st.warning("Run full analysis first."); return
    cols = st.columns(4)
    safe_metric(cols[0], "Stock Value", fmt_money(numcol(df,"stock_value").sum()))
    safe_metric(cols[1], "Avg Cover", f"{numcol(df,'days_of_cover').mean():.0f}d")
    safe_metric(cols[2], "Needs Reorder", f"{int((numcol(df,'inventory_level')<numcol(df,'reorder_point')).sum()):,}")
    safe_metric(cols[3], "Discount", f"{int((df.get('inventory_action',pd.Series(dtype=str))=='discount').sum()):,}")
    fig = safe_chart(viz.create_stock_status_pie, df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    cols = ["product_name","category","inventory_level","sales_volume","days_of_cover","stock_status","inventory_action","price_adjustment_pct","reorder_point","recommended_order_qty"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(df[available_cols].sort_values("days_of_cover"))


def render_risk_explainability() -> None:
    page_hdr("Risk & Explainability", "Multi-factor risk scoring with product-level rationale.")
    df = require_data(prefer_analyzed=True)
    if df is None or "composite_risk_score" not in df.columns: st.warning("Run full analysis first."); return
    cols = st.columns(4)
    safe_metric(cols[0], "Avg Risk", f"{numcol(df,'composite_risk_score').mean():.1f}/100")
    safe_metric(cols[1], "Critical", f"{int((df['risk_level']=='CRITICAL').sum()) if 'risk_level' in df.columns else 0:,}")
    safe_metric(cols[2], "High", f"{int((df['risk_level']=='HIGH').sum()) if 'risk_level' in df.columns else 0:,}")
    safe_metric(cols[3], "Explained", f"{int(df.get('explanation_text',pd.Series(dtype=str)).notna().sum()):,}")
    st.markdown("<div class='sub-title'>Product Explanation</div>", unsafe_allow_html=True)
    labels = df["product_name"].astype(str).tolist() if "product_name" in df.columns else [str(i) for i in df.index]
    sel = st.selectbox("Select product", labels)
    row = df[df["product_name"].astype(str) == sel].iloc[0] if "product_name" in df.columns else df.iloc[labels.index(sel)]
    kc = st.columns(4)
    kc[0].metric("Price", fmt_money(row.get("current_price",0)))
    kc[1].metric("Optimal", fmt_money(row.get("optimal_price",row.get("current_price",0))))
    kc[2].metric("Risk", f"{row.get('composite_risk_score',0):.1f}")
    kc[3].metric("Recommendation", str(row.get("recommendation","N/A")))
    st.info(str(row.get("explanation_summary","No explanation.")))
    with st.expander("Full explanation", expanded=True):
        st.write(str(row.get("explanation_text",row.get("risk_reason",""))))


# ═══════════════════════════════════════════════════════════════════════════
# REPORTS TAB
# ═══════════════════════════════════════════════════════════════════════════

def render_reports() -> None:
    page_hdr("Reports", "Generate CSV, Excel, PDF, and focused risk reports.")
    df = require_data(prefer_analyzed=True)
    if df is None: return
    ins = st.session_state.insights or build_insights(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("Download CSV", data=df_to_csv(df), file_name="pricing_analysis.csv", mime="text/csv", use_container_width=True)
    if not REPORTING_AVAILABLE:
        st.warning(f"Excel/PDF unavailable: {REPORTING_IMPORT_ERROR}")
        st.info("Install openpyxl and reportlab.")
        if "category" in df.columns:
            show_table(df.groupby("category").agg(
                products=("product_name","count"), avg_price=("current_price","mean"),
                avg_margin=("profit_margin","mean"), revenue=("revenue","sum")).reset_index())
        return
    gen = ReportGenerator(st.session_state.config)
    with c2:
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Generating..."): st.session_state["re_xlsx"] = gen.generate_excel_report(df, ins)
        if "re_xlsx" in st.session_state:
            with open(st.session_state["re_xlsx"],"rb") as f: st.download_button("Download Excel", f.read(), Path(st.session_state["re_xlsx"]).name, use_container_width=True)
    with c3:
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Generating..."): st.session_state["re_pdf"] = gen.generate_pdf_report(df, ins)
        if "re_pdf" in st.session_state:
            with open(st.session_state["re_pdf"],"rb") as f: st.download_button("Download PDF", f.read(), Path(st.session_state["re_pdf"]).name, mime="application/pdf", use_container_width=True)
    with c4:
        if st.button("Risk Report", use_container_width=True):
            with st.spinner("Generating..."): st.session_state["re_risk"] = gen.generate_risk_report(df)
        if "re_risk" in st.session_state:
            with open(st.session_state["re_risk"],"rb") as f: st.download_button("Download Risk", f.read(), Path(st.session_state["re_risk"]).name, use_container_width=True)
    if "category" in df.columns:
        st.subheader("Preview")
        show_table(df.groupby("category").agg(
            products=("product_name","count"), avg_price=("current_price","mean"),
            avg_margin=("profit_margin","mean"), revenue=("revenue","sum")).reset_index())


# ═══════════════════════════════════════════════════════════════════════════
# CHAT ASSISTANT  —  Fixed rendering with proper response formatting
# ═══════════════════════════════════════════════════════════════════════════

def _render_chat_message(msg: Dict[str, Any]) -> None:
    """Render a single chat message with proper markdown formatting."""
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if content:
            st.markdown(content, unsafe_allow_html=False)
        else:
            st.caption("*(empty response)*")

def render_chat_assistant() -> None:
    st.markdown("<div class='sec-title'>AI Pricing Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-desc'>Ask about your pricing, risk, inventory, and competitors.</div>", unsafe_allow_html=True)
    df = require_data(prefer_analyzed=True)
    if df is None: return
    if st.session_state.analyzed_df is None:
        st.warning("Run full analysis first.")
        return
    asst: AIAssistant = st.session_state.assistant
    if not getattr(asst, "_context_built", False):
        asst.load_context(df, st.session_state.insights or build_insights(df), st.session_state.alerts)

    # Quick questions as expandable shortcuts
    with st.expander("Quick Questions", expanded=False):
        qs = ["Which products are high risk?", "Where should we increase price?",
              "Which products have inventory issues?", "Summarize competitor pressure.",
              "What are the biggest margin opportunities?", "Show the most important alerts."]
        qcols = st.columns(3)
        for i, q in enumerate(qs):
            if qcols[i % 3].button(q, key=f"qq_{i}", use_container_width=True):
                ans = asst.ask(q)
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()

    # Render chat history
    for msg in st.session_state.chat_history:
        _render_chat_message(msg)

    # Chat input
    q = st.chat_input("Ask about pricing, demand, risk, inventory, competitors...")
    if q:
        ans = asst.ask(q)
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()

    if st.session_state.chat_history and st.button("Clear Chat"):
        st.session_state.chat_history = []
        asst.clear_history()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session_state()
    render_sidebar()
    route = st.session_state.nav
    dispatch = {
        "Executive Overview": render_executive_overview,
        "Data Intake": render_data_intake,
        "Batch Analysis": render_batch_analysis,
        "Pricing Engine": render_pricing_engine,
        "Demand Forecasting": render_demand_forecasting,
        "Competitor Intelligence": render_competitor_intelligence,
        "Inventory Optimization": render_inventory_optimization,
        "Risk & Explainability": render_risk_explainability,
        "Reports": render_reports,
        "Chat Assistant": render_chat_assistant,
    }
    dispatch.get(route, lambda: None)()

if __name__ == "__main__":
    main()