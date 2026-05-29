"""
AI Pricing Intelligence Platform — Professional Enterprise Dashboard

Complete Streamlit enterprise application that integrates data intake,
batch pricing analysis, forecasting, competitive intelligence, inventory
optimization, risk scoring, explainability, dashboards, reporting, and chat.
"""

from __future__ import annotations

import io
import os
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
from modules.notification_manager import (
    load_env_smtp_settings,
    send_email_alert,
)
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
    "Notifications & Alerts",
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
            padding: 2.5rem 1.65rem 1.25rem !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        .main .block-container { overflow: visible !important; }
        
        /* Ensure top spacing to prevent Streamlit header overlap */
        div[data-testid="stAppViewContainer"] {
            padding-top: 20px !important;
        }

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
            padding: 16px 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
            height: 100%;
            width: 100%;
            overflow: hidden;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-color: #BFDBFE;
            transform: translateY(-1px);
        }
        div[data-testid="stMetric"] label {
            color: var(--slate-500); font-size: 0.65rem; font-weight: 800;
            text-transform: uppercase; letter-spacing: 0.06em;
            margin-bottom: 6px;
            display: block;
            word-wrap: break-word;
            overflow: visible;
            white-space: normal;
        }
        div[data-testid="stMetric"] > div:first-child {
            font-weight: 800; color: var(--slate-950); font-size: 1.25rem !important;
            word-wrap: break-word;
            overflow: visible;
            white-space: normal;
            text-overflow: clip;
        }
        div[data-testid="stMetric"] > div:nth-child(2) {
            font-size: 0.75rem !important;
            word-wrap: break-word;
            overflow: visible;
            white-space: normal;
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
        .stPlotlyChart > div { width: 100% !important; }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.85rem;
            align-items: stretch;
        }

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
        "price_validation_send_email": True,
        "notification_history": [],
        "price_validation_results": None,
        "price_validation_summary": {},
        "summary_report": None,
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
    if df is None or df.empty:
        st.warning("No data available.")
        return
    st.session_state.raw_df = st.session_state.processed_df = df.copy()
    st.session_state.analyzed_df = None
    st.session_state.processor_report = rpt
    st.session_state.analysis_reports = {}
    st.session_state.insights = build_insights(df)
    st.session_state.alerts = []
    st.session_state.re_pdf = None
    st.session_state.summary_report = None
    st.session_state.uploaded_name = src
    st.session_state.chat_history = []
    st.session_state.assistant.clear_history()

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in ["current_price", "cost_price", "competitor_price", "sales_volume", "inventory_level", "demand_trend", "price_elasticity"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "competitor_price" not in out.columns and "current_price" in out.columns:
        out["competitor_price"] = out["current_price"] * 1.05
    for c, d in [("sales_volume", 100), ("inventory_level", 500), ("demand_trend", 0.5), ("price_elasticity", -1.5)]:
        if c not in out.columns:
            out[c] = d
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(d)
    if "category" not in out.columns: out["category"] = "Uncategorized"
    else: out["category"] = out["category"].fillna("Uncategorized").astype(str)
    if "product_name" not in out.columns: out["product_name"] = out.get("product_id", pd.Series(range(len(out)), index=out.index)).astype(str)
    else: out["product_name"] = out["product_name"].fillna("").astype(str)
    if "profit_margin" not in out.columns and {"current_price", "cost_price"}.issubset(out.columns):
        denom = out["current_price"].replace(0, pd.NA)
        out["profit_margin"] = ((out["current_price"] - out["cost_price"]) / denom).fillna(0).clip(lower=0)
    if "revenue" not in out.columns and {"current_price", "sales_volume"}.issubset(out.columns):
        out["revenue"] = out["current_price"] * out["sales_volume"]
    if "price_vs_competitor" not in out.columns and {"current_price", "competitor_price"}.issubset(out.columns):
        out["price_vs_competitor"] = (out["current_price"] / out["competitor_price"].replace(0, pd.NA)).fillna(1.0)
    return out

def numcol(df: pd.DataFrame, c: str, d: float = 0.0) -> pd.Series:
    if df is None:
        return pd.Series(dtype="float64")
    if c not in df.columns: return pd.Series([d] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[c], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(d)

def safe_sort(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out[col]):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    else:
        numeric = pd.to_numeric(out[col], errors="coerce")
        out[col] = numeric.fillna(0) if numeric.notna().any() else out[col].fillna("").astype(str)
    return out.sort_values(col, ascending=ascending)

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


def _get_notification_config() -> Dict[str, Any]:
    env = load_env_smtp_settings()
    required_env = ["SMTP_SERVER", "SMTP_PORT", "SMTP_EMAIL", "SMTP_PASSWORD", "RECIPIENT_EMAIL"]
    try:
        smtp_port = int(env["smtp_port"])
    except (ValueError, TypeError):
        smtp_port = 587

    return {
        "recipient": env["recipient_email"],
        "sender": env["smtp_email"],
        "smtp_server": env["smtp_server"],
        "smtp_port": smtp_port,
        "smtp_password": env["smtp_password"],
        "configured": all(bool(os.getenv(name, "").strip()) for name in required_env),
    }


def _record_notification_history(alert_type: str, status: str, recipient: str, note: str) -> None:
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Alert Type": alert_type,
        "Status": status,
        "Recipient": recipient,
        "Note": note,
    }
    history = st.session_state.notification_history or []
    history.insert(0, entry)
    st.session_state.notification_history = history[:25]


def build_summary_report_text(df: pd.DataFrame, insights: Dict[str, Any]) -> str:
    def safe_percentage(part: float, total: float) -> Optional[float]:
        if total is None or total <= 0:
            return None
        return max(0.0, min(100.0, float(part) / float(total) * 100.0))

    dataset_name = getattr(st.session_state, "uploaded_name", "Uploaded dataset") or "Uploaded dataset"
    total_products = int(insights.get("total_products", len(df)))

    revenue_available = "revenue" in df.columns
    revenue_series = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0) if revenue_available else pd.Series(dtype="float64")
    total_revenue = float(revenue_series.sum()) if revenue_available else 0.0
    revenue_valid = revenue_available and total_revenue > 0

    recommendation_available = "recommendation" in df.columns
    rec_counts = df["recommendation"].value_counts() if recommendation_available else pd.Series(dtype="int64")
    price_increase_count = int(rec_counts.get("Increase", 0))
    price_reduce_count = int(rec_counts.get("Decrease", 0))
    maintain_count = int(rec_counts.get("Maintain", 0))

    demand_counts = {}
    if "demand_trend_category" in df.columns:
        demand_counts = df["demand_trend_category"].fillna("Stable").value_counts().to_dict()
    elif "demand_trend" in df.columns:
        demand_counts = {
            "Increasing": int((df["demand_trend"] > 0.55).sum()),
            "Stable": int(((df["demand_trend"] >= 0.45) & (df["demand_trend"] <= 0.55)).sum()),
            "Declining": int((df["demand_trend"] < 0.45).sum()),
        }

    inventory_counts = {}
    if "stock_status" in df.columns:
        inventory_counts = df["stock_status"].fillna("Balanced").value_counts().to_dict()
    elif "inventory_action" in df.columns:
        inventory_counts = df["inventory_action"].fillna("Balanced").value_counts().to_dict()

    risk_available = "risk_level" in df.columns
    low_risk = int(df["risk_level"].isin(["LOW"]).sum()) if risk_available else 0
    moderate_risk = int(df["risk_level"].isin(["MEDIUM"]).sum()) if risk_available else 0
    high_risk = int(df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()) if risk_available else 0

    category_available = "category" in df.columns and revenue_valid
    category_summary = []
    if category_available:
        cat_revenue = df.groupby("category")["revenue"].sum().clip(lower=0.0)
        cat_revenue = cat_revenue[cat_revenue > 0.0]
        total_category_revenue = float(cat_revenue.sum())
        if total_category_revenue > 0:
            for category, value in cat_revenue.sort_values(ascending=False).head(3).items():
                share = safe_percentage(value, total_revenue)
                if share is not None:
                    category_summary.append((category, int(df[df["category"] == category].shape[0]), share))

    top_product_segment = None
    if revenue_valid and "product_name" in df.columns:
        best = revenue_series.nlargest(1)
        if not best.empty:
            idx = best.index[0]
            top_share = safe_percentage(best.iloc[0], total_revenue)
            if top_share is not None:
                top_product_name = str(df.at[idx, "product_name"] or df.at[idx, "product_id"] if "product_id" in df.columns else "Top product")
                top_product_segment = (top_product_name, top_share)

    competitor_summary = {}
    if "overpriced_flag" in df.columns or "underpriced_flag" in df.columns:
        overpriced = int(df.get("overpriced_flag", pd.Series(dtype=bool)).fillna(False).sum()) if "overpriced_flag" in df.columns else 0
        underpriced = int(df.get("underpriced_flag", pd.Series(dtype=bool)).fillna(False).sum()) if "underpriced_flag" in df.columns else 0
        competitor_summary = {
            "Products priced above market benchmark": overpriced,
            "Products priced below market benchmark": underpriced,
        }

    lines = [
        "AI Pricing Intelligence Report",
        "",
        f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {dataset_name}",
        "",
        "EXECUTIVE SUMMARY",
        "This analysis evaluated pricing, demand, inventory, competitor positioning, and risk indicators across the product portfolio.",
        "Key observations:",
    ]

    if price_increase_count:
        lines.append(f"• {price_increase_count:,} products are recommended for price increase.")
    if price_reduce_count:
        lines.append(f"• {price_reduce_count:,} products are recommended for price decrease.")
    if maintain_count:
        lines.append(f"• {maintain_count:,} products should maintain current pricing.")
    if demand_counts:
        increasing = demand_counts.get("Increasing", demand_counts.get("increasing", 0))
        stable = demand_counts.get("Stable", demand_counts.get("stable", 0))
        declining = demand_counts.get("Declining", demand_counts.get("declining", 0))
        if increasing:
            lines.append(f"• {increasing:,} products show improving demand trends.")
        if stable:
            lines.append(f"• {stable:,} products have stable demand profiles.")
        if declining:
            lines.append(f"• {declining:,} products show declining demand signals.")
    if inventory_counts:
        if inventory_counts.get("Overstock", inventory_counts.get("overstock", 0)):
            lines.append(f"• Overstock risk is present in {inventory_counts.get('Overstock', inventory_counts.get('overstock', 0)):,} products.")
        if inventory_counts.get("Understock", inventory_counts.get("understock", 0)):
            lines.append(f"• Understock risk is present in {inventory_counts.get('Understock', inventory_counts.get('understock', 0)):,} products.")
    if risk_available:
        lines.append(f"• Portfolio risk includes low, moderate, and high risk product segments.")
    lines.extend([
        "",
        "BUSINESS INSIGHTS",
        "Pricing Actions",
        f"- Products recommended for price increase: {price_increase_count:,}",
        f"- Products recommended for price decrease: {price_reduce_count:,}",
        f"- Products recommended to maintain pricing: {maintain_count:,}",
        "",
        "Revenue Observations",
    ])

    if category_summary:
        for category, count, share in category_summary:
            lines.append(f"- {category}: {count:,} products, representing {share:.0f}% of revenue.")
        if top_product_segment:
            lines.append(f"- Top product '{top_product_segment[0]}' contributes {top_product_segment[1]:.0f}% of portfolio revenue.")
        if len(category_summary) and category_summary[0][2] > 50:
            lines.append("- The portfolio is concentrated in a single category, which may require focused performance monitoring.")
        else:
            lines.append("- Revenue is distributed across categories, supporting a balanced portfolio view.")
    else:
        lines.append("- Selected insight unavailable for current dataset.")

    lines.extend([
        "",
        "Risk Overview",
    ])
    if risk_available:
        lines.append(f"- Low risk products: {low_risk:,}")
        lines.append(f"- Moderate risk products: {moderate_risk:,}")
        lines.append(f"- High risk products: {high_risk:,}")
        if high_risk > 0:
            lines.append("- Major attention is recommended for products with high risk exposure.")
        else:
            lines.append("- Risk exposure is currently manageable across the portfolio.")
    else:
        lines.append("- Selected insight unavailable for current dataset.")

    lines.extend([
        "",
        "Inventory & Demand Signals",
    ])

    if inventory_counts or demand_counts:
        if inventory_counts:
            if inventory_counts.get("Overstock", inventory_counts.get("overstock", 0)):
                lines.append(f"- Overstock observations: {inventory_counts.get('Overstock', inventory_counts.get('overstock', 0)):,} products.")
            if inventory_counts.get("Understock", inventory_counts.get("understock", 0)):
                lines.append(f"- Understock observations: {inventory_counts.get('Understock', inventory_counts.get('understock', 0)):,} products.")
            if inventory_counts.get("Balanced", inventory_counts.get("balanced", 0)):
                lines.append(f"- Balanced inventory is present in {inventory_counts.get('Balanced', inventory_counts.get('balanced', 0)):,} products.")
        if demand_counts:
            increasing = demand_counts.get("Increasing", demand_counts.get("increasing", 0))
            stable = demand_counts.get("Stable", demand_counts.get("stable", 0))
            declining = demand_counts.get("Declining", demand_counts.get("declining", 0))
            if increasing:
                lines.append(f"- Increasing demand trends are observed in {increasing:,} products.")
            if stable:
                lines.append(f"- Stable demand trends are observed in {stable:,} products.")
            if declining:
                lines.append(f"- Declining demand signals are observed in {declining:,} products.")
    else:
        lines.append("- Selected insight unavailable for current dataset.")

    if competitor_summary:
        lines.extend([
            "",
            "COMPETITOR ANALYSIS",
        ])
        for label, count in competitor_summary.items():
            lines.append(f"- {label}: {count:,}")
        lines.append("- Competitor positioning highlights where pricing adjustments can improve market alignment.")

    lines.extend([
        "",
        "AI STRATEGIC RECOMMENDATIONS",
        "Priority Actions:",
        "1. Optimize pricing on products with strong demand and potential margin improvement.",
        "2. Adjust inventory levels to reduce holding cost and prevent stock gaps.",
        "3. Focus on high-potential products where demand is trending upward.",
        "4. Review competitor-sensitive products for pricing alignment.",
        "5. Monitor future pricing refresh alerts to maintain competitiveness.",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_product_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _format_price_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _format_difference_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):+,.2f}"
    except (TypeError, ValueError):
        return "-"


def _load_price_refresh_file(uploaded_file: Any) -> tuple[Optional[pd.DataFrame], str]:
    if uploaded_file is None:
        return None, "Upload a refreshed price file first."
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        return None, "Invalid file format. Upload a CSV or XLSX file."
    try:
        raw = uploaded_file.getvalue()
        if not raw:
            return None, "Uploaded file is empty."
        if suffix == ".xlsx":
            return pd.read_excel(io.BytesIO(raw), engine="openpyxl"), ""
        return pd.read_csv(io.BytesIO(raw)), ""
    except Exception as exc:
        logger.exception("Price refresh file load failed")
        return None, f"Unable to read refreshed price file: {exc}"


def _prepare_price_validation_frame(
    df: pd.DataFrame,
    source: str,
    price_column: str,
    use_product_id: bool,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    required = {"product_name", price_column}
    if df is None or df.empty:
        return pd.DataFrame(), [f"{source} file is empty."]
    missing = sorted(required - set(df.columns))
    if missing:
        return pd.DataFrame(), [f"{source} file is missing required columns: {', '.join(missing)}."]

    work = df.copy()
    if "product_id" not in work.columns:
        work["product_id"] = pd.NA
    work = work[["product_id", "product_name", price_column]].copy()
    work["product_id"] = work["product_id"].astype("string").fillna("").str.strip()
    work["product_name"] = work["product_name"].astype("string").fillna("").str.strip()
    work["name_key"] = work["product_name"].map(_normalize_product_name)
    work[price_column] = pd.to_numeric(work[price_column], errors="coerce")

    invalid_names = int((work["name_key"] == "").sum())
    invalid_prices = int(work[price_column].isna().sum())
    if invalid_names:
        warnings.append(f"{source} file has {invalid_names} row(s) without product_name; those rows were ignored.")
    if invalid_prices:
        warnings.append(f"{source} file has {invalid_prices} row(s) with invalid {price_column}; those rows were ignored.")
    work = work[(work["name_key"] != "") & work[price_column].notna()].copy()
    if work.empty:
        return pd.DataFrame(), warnings + [f"{source} file has no valid product rows."]
    work["_source_row"] = work.index

    if use_product_id:
        work["_match_key"] = work["product_id"].where(work["product_id"] != "", "name:" + work["name_key"])
        work["_match_key"] = work["_match_key"].where(work["product_id"] == "", "id:" + work["product_id"])
    else:
        work["_match_key"] = "name:" + work["name_key"]
    duplicate_count = int(work["_match_key"].duplicated(keep="last").sum())
    if duplicate_count:
        warnings.append(f"{source} file has {duplicate_count} duplicate product key(s); the last row was used.")
    return work.drop_duplicates("_match_key", keep="last"), warnings


def compare_price_files(analysis_df: pd.DataFrame, refresh_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int], list[str]]:
    use_product_id = "product_id" in analysis_df.columns and "product_id" in refresh_df.columns
    original, original_warnings = _prepare_price_validation_frame(analysis_df, "Analysis dataset", "current_price", use_product_id)
    refreshed, refresh_warnings = _prepare_price_validation_frame(refresh_df, "Refreshed price", "custom_price", use_product_id)
    warnings = original_warnings + refresh_warnings
    if original.empty or refreshed.empty:
        return pd.DataFrame(), {}, warnings

    if use_product_id:
        original_with_id = original[original["product_id"] != ""]
        refreshed_with_id = refreshed[refreshed["product_id"] != ""]
        id_matches = original_with_id.merge(
            refreshed_with_id,
            on="product_id",
            how="inner",
            suffixes=("_original", "_refreshed"),
        )
        original_remaining = original[~original["_source_row"].isin(id_matches["_source_row_original"])]
        refreshed_remaining = refreshed[~refreshed["_source_row"].isin(id_matches["_source_row_refreshed"])]
        name_matches = original_remaining.merge(
            refreshed_remaining,
            on="name_key",
            how="outer",
            suffixes=("_original", "_refreshed"),
            indicator=True,
        )
        id_matches["_merge"] = "both"
        matched = pd.concat([id_matches, name_matches], ignore_index=True, sort=False)
        matched["product_id_original"] = matched.get("product_id_original", matched.get("product_id", pd.Series(dtype="string"))).fillna(matched.get("product_id", ""))
        matched["product_id_refreshed"] = matched.get("product_id_refreshed", matched.get("product_id", pd.Series(dtype="string"))).fillna(matched.get("product_id", ""))
    else:
        matched = original.merge(
            refreshed,
            on="_match_key",
            how="outer",
            suffixes=("_original", "_refreshed"),
            indicator=True,
        )
    matched["product_id"] = matched["product_id_original"].fillna(matched["product_id_refreshed"])
    matched["product_name"] = matched["product_name_original"].fillna(matched["product_name_refreshed"])
    matched["current_price"] = matched["current_price"]
    matched["custom_price"] = matched["custom_price"]
    matched["difference"] = matched["custom_price"] - matched["current_price"]

    matched["status"] = "NO CHANGE"
    changed_mask = (matched["_merge"] == "both") & (matched["difference"].abs() > 0.000001)
    matched.loc[changed_mask, "status"] = "PRICE CHANGED"
    matched.loc[matched["_merge"] == "left_only", "status"] = "MISSING IN REFRESH FILE"
    matched.loc[matched["_merge"] == "right_only", "status"] = "NEW PRODUCT"

    result = matched[["product_id", "product_name", "current_price", "custom_price", "difference", "status"]].copy()
    result["product_id"] = result["product_id"].fillna("").astype(str)
    result["product_name"] = result["product_name"].fillna("Unknown").astype(str)
    status_counts = result["status"].value_counts()
    summary = {
        "total_compared": int(len(result)),
        "matching_prices": int(status_counts.get("NO CHANGE", 0)),
        "price_changed_count": int(status_counts.get("PRICE CHANGED", 0)),
        "missing_products": int(status_counts.get("MISSING IN REFRESH FILE", 0)),
        "new_products": int(status_counts.get("NEW PRODUCT", 0)),
    }
    return result.sort_values(["status", "product_name"]).reset_index(drop=True), summary, warnings


def build_alert_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    return results_df[results_df["status"] == "PRICE CHANGED"].copy()


def detect_price_mismatches(results_df: pd.DataFrame) -> pd.DataFrame:
    return build_alert_dataframe(results_df)


def _build_price_validation_email_body(summary: Dict[str, int], mismatch_df: pd.DataFrame, source_name: str) -> str:
    changed_count = summary.get("price_changed_count", 0)
    lines = [
        f"{changed_count:,} price changes detected.",
        "",
        "Changed Products:",
    ]
    if mismatch_df is None or mismatch_df.empty:
        lines.append("No mismatches detected.")
    else:
        for _, row in mismatch_df.head(8).iterrows():
            lines.append(
                f"• {row.get('product_name', 'Unknown')} — "
                f"Old: {_format_price_value(row.get('current_price'))} → "
                f"New: {_format_price_value(row.get('custom_price'))}"
            )
        if len(mismatch_df) > 8:
            lines.append(f"Additional changed products: {len(mismatch_df) - 8:,}. See attached CSV.")
    lines.extend([
        "",
        "Analysis Dataset:",
        source_name or "Loaded dataset",
        "",
        "Timestamp:",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ])
    return "\n".join(lines)


def send_price_validation_email(results_df: pd.DataFrame, summary: Dict[str, int], source_name: str) -> tuple[bool, str]:
    mismatch_df = build_alert_dataframe(results_df)
    should_send = summary.get("price_changed_count", 0) > 0
    cfg = _get_notification_config()
    if not should_send:
        _record_notification_history("Price Refresh Validation", "Skipped", cfg.get("recipient", ""), "No mismatches detected")
        return False, "No mismatches detected."
    if not cfg["configured"] or not cfg["smtp_server"] or not cfg["smtp_port"] or not cfg["sender"] or not cfg["recipient"] or not cfg["smtp_password"]:
        _record_notification_history("Price Refresh Validation", "Skipped", cfg.get("recipient", ""), "SMTP settings incomplete")
        return False, "SMTP configuration missing in .env. Email notification skipped."

    success, status = send_email_alert(
        "Price Refresh Validation Alert – Price Changes Detected",
        _build_price_validation_email_body(summary, mismatch_df, source_name),
        cfg["recipient"],
        cfg["sender"],
        cfg["smtp_server"],
        cfg["smtp_port"],
        cfg["smtp_password"],
        attachments=[("price_refresh_validation_results.csv", results_df.to_csv(index=False).encode("utf-8"), "text/csv")],
    )
    _record_notification_history("Price Refresh Validation", "Sent" if success else "Failed", cfg["recipient"], status)
    return success, status


def run_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    pr = st.progress(0, text="Preparing products")
    rpts: Dict[str, Any] = {}
    w = ensure_cols(df)
    if w.empty:
        pr.empty()
        raise ValueError("No data available for analysis.")
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
        if fig is None:
            return None
        
        # Validate figure has actual data (not just empty annotations)
        if hasattr(fig, 'data') and len(fig.data) > 0:
            # Check that data traces actually have values
            has_valid_data = False
            for trace in fig.data:
                # Check for x/y values (bar, scatter, line)
                if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
                    has_valid_data = True
                    break
                if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
                    has_valid_data = True
                    break
                # Check for values (pie/donut charts)
                if hasattr(trace, 'values') and trace.values is not None and len(trace.values) > 0:
                    has_valid_data = True
                    break
                # Check for z values (heatmaps)
                if hasattr(trace, 'z') and trace.z is not None and (len(trace.z) > 0 if not isinstance(trace.z, (int, float)) else True):
                    has_valid_data = True
                    break
            if has_valid_data:
                return fig
        
        return None
    except Exception as exc:
        logger.exception("Chart build failed: %s", getattr(fn, "__name__", fn))
        return None

def render_plotly(fig: Optional[go.Figure]) -> None:
    if fig is None:
        st.caption("No data available for this chart yet. Run analysis first.")
        return
    try:
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True, "displayModeBar": False})
    except Exception as exc:
        logger.exception("Plotly render failed")
        st.caption("Error rendering chart. Please try again.")

def show_table(df_slice: pd.DataFrame) -> None:
    """Render a clean table — safe with empty/None guards, removes empty rows."""
    if df_slice is None or not isinstance(df_slice, pd.DataFrame) or df_slice.empty:
        st.caption("No data available.")
        return
    
    safe_df = df_slice.copy()
    # Replace inf with NA
    safe_df = safe_df.replace([float("inf"), float("-inf")], pd.NA)
    
    # Remove rows that are completely empty (all NaN)
    safe_df = safe_df.dropna(how="all")
    
    if safe_df.empty:
        st.caption("No valid data to display.")
        return
    
    height = min(620, max(240, 38 * (len(safe_df) + 1)))
    try:
        st.dataframe(safe_df, use_container_width=True, hide_index=True, height=height)
    except TypeError:
        try:
            st.dataframe(safe_df, use_container_width=True, height=height)
        except Exception as exc:
            logger.exception("Table render error")
            st.caption("Error rendering table.")
    except Exception as exc:
        logger.exception("Table render error")
        st.caption("Error rendering table.")

def category_preview(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "category" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    if "product_name" not in work.columns:
        work["product_name"] = work.index.astype(str)
    for col in ["current_price", "profit_margin", "revenue"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    return work.groupby("category", dropna=False).agg(
        products=("product_name", "count"),
        avg_price=("current_price", "mean"),
        avg_margin=("profit_margin", "mean"),
        revenue=("revenue", "sum"),
    ).reset_index()

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
        for k in ["raw_df","processed_df","analyzed_df","processor_report","analysis_reports","insights","alerts","chat_history","re_pdf","summary_report","uploaded_name","last_run_at"]:
            st.session_state[k] = [] if k in {"alerts","chat_history"} else ({} if k in {"analysis_reports","insights"} else None)
        st.session_state.assistant.clear_history(); st.rerun()
    st.sidebar.divider()
    st.sidebar.markdown("<div class='small-muted'>v2.0 · AI Pricing Platform</div>", unsafe_allow_html=True)


def require_data(prefer_analyzed: bool = False) -> Optional[pd.DataFrame]:
    df = get_df(prefer_analyzed=prefer_analyzed)
    if df is None: st.info("Upload a CSV/Excel or load the sample dataset from the sidebar.")
    elif df.empty:
        st.warning("No data available.")
        return None
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TABS  —  All use safe rendering patterns, no raw HTML div wrapping Streamlit elements
# ═══════════════════════════════════════════════════════════════════════════

def render_executive_overview() -> None:
    page_hdr("Executive Overview", "Portfolio-level pricing, risk, demand, and margin signals.")
    df = require_data(prefer_analyzed=True)
    if df is None: return
    ins = st.session_state.insights or build_insights(df)

    # KPI Metrics - improved spacing and width for full number display
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    
    safe_metric(cols[0], "Products", f"{ins.get('total_products',0):,}")
    safe_metric(cols[1], "Projected Revenue", fmt_money(ins.get("expected_revenue",0)))
    safe_metric(cols[2], "Margin", f"{ins.get('avg_margin',0)*100:.1f}%")
    safe_metric(cols[3], "High Risk", f"{ins.get('high_risk_count',0):,}")

    # Charts section
    st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    l, r = st.columns([1.25, 1])
    with l:
        fig = safe_chart(viz.create_revenue_by_category_chart, df)
        if fig:
            render_plotly(fig)
        else:
            st.caption("Revenue chart not available yet. Run analysis first.")
    with r:
        fig = None
        if "recommendation" in df.columns:
            fig = safe_chart(viz.create_recommendation_pie, df)
        if fig is None:
            margin_col = "profit_margin" if "profit_margin" in df.columns else "margin_percentage"
            fig = safe_chart(viz.create_margin_distribution_chart, df, margin_col=margin_col)
        if fig:
            render_plotly(fig)
        else:
            st.caption("Distribution chart not available yet. Run analysis first.")

    st.markdown("<div class='sub-title'>Priority Actions</div>", unsafe_allow_html=True)
    pcols = ["product_id","product_name","category","current_price","optimal_price","price_change_pct","recommendation","composite_risk_score","risk_level"]
    av = [c for c in pcols if c in df.columns]
    if av:
        sc = "composite_risk_score" if "composite_risk_score" in df.columns else av[0]
        sorted_df = safe_sort(df, sc, ascending=False)[av].head(15)
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
    er = numcol(df, "expected_revenue").sum()
    ep = numcol(df, "expected_profit").sum()
    avg_margin_val = float(df.get("profit_margin", pd.Series([0])).mean()) * 100
    high_risk_count = int((df.get("risk_level", pd.Series(dtype=str)).isin(["HIGH", "CRITICAL"])).sum()) if "risk_level" in df.columns else 0
    inc_count = int((df.get("recommendation", pd.Series(dtype=str)) == "Increase").sum()) if "recommendation" in df.columns else 0
    cols = st.columns(4)
    safe_metric(cols[0], "Projected Revenue", fmt_money(er))
    safe_metric(cols[1], "Profit", fmt_money(ep))
    safe_metric(cols[2], "Avg Margin", f"{avg_margin_val:.1f}%")
    safe_metric(cols[3], "High Risk", f"{high_risk_count:,}")
    l, r = st.columns(2)
    for col, fn in [(l, viz.create_price_vs_optimal_scatter), (r, viz.create_recommendation_pie)]:
        with col:
            try:
                fig = safe_chart(fn, df)
                if fig:
                    render_plotly(fig)
                else:
                    st.caption("No data available for chart.")
            except Exception:
                st.caption("Chart unavailable.")
    cols = ["product_name","category","current_price","optimal_price","price_change_pct","recommendation","expected_revenue","expected_profit","explanation_summary"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        sorted_df = safe_sort(df[available_cols], "price_change_pct", ascending=False)
        show_table(sorted_df)


def render_demand_forecasting() -> None:
    page_hdr("Demand Forecasting", "30-day projections and trend segmentation.")
    df = require_data(prefer_analyzed=True)
    if df is None or "predicted_demand" not in df.columns: 
        st.warning("Run full analysis first.")
        return
    
    # KPI Metrics
    cols = st.columns(4)
    safe_metric(cols[0], "Avg Demand", f"{numcol(df,'predicted_demand').mean():,.0f}")
    safe_metric(cols[1], "Confidence", f"{numcol(df,'forecast_confidence').mean()*100:.0f}%")
    safe_metric(cols[2], "Total Forecast", f"{numcol(df,'forecast_next_30d').sum():,.0f}")
    inc = int((df.get("demand_trend_category", pd.Series(dtype=str)) == "increasing").sum())
    safe_metric(cols[3], "Increasing", f"{inc:,}")

    st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Charts section
    l, r = st.columns(2, gap="medium")
    
    with l:
        st.markdown("<div class='sub-title' style='margin-top: 0; margin-bottom: 0.5rem;'>Weekly Forecast</div>", unsafe_allow_html=True)
        fig = safe_chart(viz.create_weekly_forecast_chart, df)
        if fig:
            render_plotly(fig)
        else:
            st.warning("Weekly forecast chart unavailable. Check data for forecast_week columns.")
    
    with r:
        st.markdown("<div class='sub-title' style='margin-top: 0; margin-bottom: 0.5rem;'>Trend Distribution</div>", unsafe_allow_html=True)
        fig = safe_chart(viz.create_trend_distribution_chart, df)
        if fig:
            render_plotly(fig)
        else:
            st.warning("Trend distribution chart unavailable. Check data for demand_trend_category column.")
    
    # Data table
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Forecast Details</div>", unsafe_allow_html=True)
    
    cols = ["product_name","category","sales_volume","predicted_demand","forecast_confidence","forecast_ci_lower","forecast_ci_upper","demand_trend_category"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(safe_sort(df[available_cols], "predicted_demand", ascending=False))
    else:
        st.warning("No forecast columns available.")


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
        render_plotly(fig)
    cols = ["product_name","category","current_price","competitor_price","market_position","price_gap_pct","competitive_score","pricing_recommendation"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(safe_sort(df[available_cols], "competitive_score"))


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
        render_plotly(fig)
    cols = ["product_name","category","inventory_level","sales_volume","days_of_cover","stock_status","inventory_action","price_adjustment_pct","reorder_point","recommended_order_qty"]
    available_cols = [c for c in cols if c in df.columns]
    if available_cols:
        show_table(safe_sort(df[available_cols], "days_of_cover"))


def render_risk_explainability() -> None:
    page_hdr("Risk & Explainability", "Multi-factor risk scoring with product-level rationale.")
    df = require_data(prefer_analyzed=True)
    if df is None or "composite_risk_score" not in df.columns: st.warning("Run full analysis first."); return
    cols = st.columns(4)
    safe_metric(cols[0], "Avg Risk", f"{numcol(df,'composite_risk_score').mean():.1f}/100")
    safe_metric(cols[1], "Critical", f"{int((df['risk_level']=='CRITICAL').sum()) if 'risk_level' in df.columns else 0:,}")
    safe_metric(cols[2], "High", f"{int((df['risk_level']=='HIGH').sum()) if 'risk_level' in df.columns else 0:,}")
    safe_metric(cols[3], "Explained", f"{int(df.get('explanation_text',pd.Series(dtype=str)).notna().sum()):,}")

    # Product selector
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    labels = df["product_name"].astype(str).tolist() if "product_name" in df.columns else [str(i) for i in df.index]
    sel = st.selectbox("Select a product to analyse", labels)
    row = df[df["product_name"].astype(str) == sel].iloc[0] if "product_name" in df.columns else df.iloc[labels.index(sel)]

    # ─── Product Summary Card ──────────────────────────────────────────
    pid = str(row.get("product_id", ""))
    pname = str(row.get("product_name", ""))
    curr_price = float(row.get("current_price", 0))
    opt_price = float(row.get("optimal_price", curr_price))
    price_chg = float(row.get("price_change_pct", 0))
    rec = str(row.get("recommendation", "N/A"))
    risk_score = float(row.get("composite_risk_score", 0))
    risk_lvl = str(row.get("risk_level", "N/A"))
    margin_val = float(row.get("profit_margin", 0)) * 100

    # Recommendation badge colour
    rec_color = "#10B981" if rec.lower() == "increase" else "#EF4444" if rec.lower() == "decrease" else "#3B82F6"
    risk_color = "#EF4444" if risk_lvl in ("HIGH", "CRITICAL") else "#F59E0B" if risk_lvl == "MEDIUM" else "#10B981"

    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:1.25rem 1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); margin-bottom:1rem;">
        <h3 style="margin:0 0 0.25rem 0; font-size:1.15rem; font-weight:700; color:#0F172A;">{pname}</h3>
        <div style="font-size:0.82rem; color:#64748B; margin-bottom:1rem;">Product ID: {pid} | Category: {str(row.get("category", "—"))}</div>
        <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
            <div><span style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Current Price</span><br><span style="font-size:1.25rem; font-weight:800; color:#0F172A;">${curr_price:,.2f}</span></div>
            <div><span style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Recommended Price</span><br><span style="font-size:1.25rem; font-weight:800; color:#0F172A;">${opt_price:,.2f}</span></div>
            <div><span style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Price Change</span><br><span style="font-size:1.25rem; font-weight:800; color:{'#EF4444' if price_chg < 0 else '#10B981'};">{price_chg:+.1f}%</span></div>
            <div><span style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Recommendation</span><br><span style="display:inline-block; margin-top:2px; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; color:#FFFFFF; background:{rec_color};">{rec}</span></div>
            <div><span style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Risk Score</span><br><span style="display:inline-block; margin-top:2px; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:700; color:#FFFFFF; background:{risk_color};">{risk_score:.1f} / 100</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Why This Recommendation? ──────────────────────────────────────
    st.markdown("<h4 style='font-size:1rem; font-weight:700; color:#0F172A; margin:0.5rem 0 0.75rem;'>Why this recommendation?</h4>", unsafe_allow_html=True)

    raw_bullets = str(row.get("explanation_bullets", "")).split("; ")
    explanation_text_raw = str(row.get("explanation_text", ""))

    # Build structured business bullet points from explanation data
    direction_map = {}
    for fname in ["demand", "inventory", "competitor", "margin", "cost"]:
        col_name = f"explanation_{fname}_direction"
        if col_name in row:
            direction_map[fname] = str(row[col_name])
        else:
            direction_map[fname] = "neutral"

    # Demand bullet
    demand_notes = {
        "upward": ("Strong customer demand detected.", "Demand levels are high, meaning the product remains competitive even after pricing adjustment."),
        "slight_upward": ("Moderate customer demand.", "Demand signals are positive, supporting current pricing strategy."),
        "downward": ("Weakening customer demand.", "Demand levels are declining, warranting a cautious pricing approach."),
        "slight_downward": ("Slightly softening demand.", "Demand trends show minor softening — monitor closely."),
        "neutral": ("Stable customer demand.", "Demand levels are consistent with market expectations."),
    }

    # Inventory bullet
    inventory_notes = {
        "upward": ("Low inventory levels detected.", "Limited stock creates scarcity value, supporting price adjustments."),
        "slight_upward": ("Adequate inventory coverage.", "Stock levels are balanced with current sales velocity."),
        "downward": ("Excess inventory accumulated.", "High stock levels relative to sales suggest promotional action may be needed."),
        "slight_downward": ("Slightly elevated inventory.", "Inventory is marginally above target — monitor turnover rates."),
        "neutral": ("Healthy inventory position.", "Stock levels are well-balanced with demand patterns."),
    }

    # Competitor bullet
    competitor_notes = {
        "upward": ("Competitive pricing advantage.", "Our price compares favourably to competitors, providing market positioning strength."),
        "slight_upward": ("Slight competitive advantage.", "Minor pricing gaps exist versus competitors — maintain current strategy."),
        "downward": ("Competitive pressure detected.", "Competitors are pricing lower, creating pressure to adjust our position."),
        "slight_downward": ("Minor competitive pressure.", "Small competitor price gaps exist — review if trends continue."),
        "neutral": ("Competitive position is stable.", "Our pricing is aligned with competitor benchmarks."),
    }

    # Margin bullet
    margin_notes = {
        "upward": ("Healthy profit margin available.", f"Current margin is strong ({margin_val:.1f}%), allowing safe pricing flexibility."),
        "slight_upward": ("Adequate profit margin.", f"Margin at {margin_val:.1f}% supports current pricing decisions."),
        "downward": ("Margin improvement needed.", f"Current margin of {margin_val:.1f}% is below target — consider cost or price adjustments."),
        "slight_downward": ("Margin requires attention.", f"Margin of {margin_val:.1f}% needs monitoring to prevent erosion."),
        "neutral": ("Profit margin is stable.", f"Margin at {margin_val:.1f}% is within acceptable range."),
    }

    # Cost bullet
    cost_notes = {
        "upward": ("Cost pressure identified.", "Rising costs relative to price require management attention to protect margins."),
        "slight_upward": ("Moderate cost structure.", "Cost ratio is within monitoring range — no immediate action needed."),
        "downward": ("Cost advantage present.", "Favourable cost structure provides competitive flexibility."),
        "slight_downward": ("Slight cost advantage.", "Cost position is marginally favourable."),
        "neutral": ("Cost structure is healthy.", "Costs are well-controlled relative to pricing."),
    }

    note_map = {
        "demand": demand_notes,
        "inventory": inventory_notes,
        "competitor": competitor_notes,
        "margin": margin_notes,
        "cost": cost_notes,
    }

    # Determine which factors to show
    factor_labels = {
        "demand": "Demand",
        "inventory": "Inventory",
        "competitor": "Competitor",
        "margin": "Margin",
        "cost": "Cost",
    }

    shown_any = False
    for fname in ["demand", "inventory", "competitor", "margin", "cost"]:
        direction = direction_map.get(fname, "neutral")
        notes = note_map.get(fname, {}).get(direction, None)
        if notes:
            shown_any = True
            title, detail = notes
            icon_map = {"upward": "▲", "slight_upward": "↗", "downward": "▼", "slight_downward": "↘", "neutral": "●"}
            icon = icon_map.get(direction, "●")
            st.markdown(f"""
            <div style="background:#FAFBFC; border-left:3px solid #3B82F6; border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem;">
                <div style="font-size:0.85rem; font-weight:700; color:#0F172A;">{icon} <span style="color:#3B82F6;">{factor_labels[fname]}:</span> {title}</div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    if not shown_any:
        st.markdown("""
        <div style="background:#FAFBFC; border-left:3px solid #94A3B8; border-radius:6px; padding:0.6rem 1rem; margin-bottom:0.5rem;">
            <div style="font-size:0.85rem; font-weight:600; color:#0F172A;">All factors are balanced.</div>
            <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">No significant drivers identified — current pricing is well-positioned.</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Key Drivers Summary ───────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size:1rem; font-weight:700; color:#0F172A; margin:0 0 0.75rem;'>Key Drivers</h4>", unsafe_allow_html=True)

    # Determine driver quality labels
    def driver_label(direction: str, factor_name: str) -> tuple:
        if direction in ("upward",):
            return ("Strong", "#10B981") if factor_name in ("demand", "margin") else ("Elevated", "#F59E0B")
        elif direction in ("slight_upward",):
            return ("Positive", "#10B981") if factor_name in ("demand",) else ("Moderate", "#3B82F6")
        elif direction in ("downward",):
            return ("Weak", "#EF4444") if factor_name in ("demand",) else ("Concerning", "#EF4444")
        elif direction in ("slight_downward",):
            return ("Softening", "#F59E0B")
        else:
            return ("Stable", "#3B82F6")

    kd1, kd2, kd3, kd4 = st.columns(4)
    for col, fname, label in [
        (kd1, "demand", "Demand Strength"),
        (kd2, "margin", "Margin Health"),
        (kd3, "competitor", "Competitive Position"),
        (kd4, "risk", "Risk Level"),
    ]:
        if col is None:
            continue
        if fname == "risk":
            lbl, clr = ("Low", "#10B981") if risk_score < 30 else ("Moderate", "#F59E0B") if risk_score < 50 else ("High", "#EF4444")
        else:
            direction = direction_map.get(fname, "neutral")
            lbl, clr = driver_label(direction, fname)
        col.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:10px; padding:0.6rem 0.8rem; text-align:center; box-shadow:0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size:0.65rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:700; margin-bottom:4px;">{label}</div>
            <div style="display:inline-block; padding:1px 10px; border-radius:999px; font-size:0.75rem; font-weight:700; color:#FFFFFF; background:{clr};">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Technical Details (collapsible) ───────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    with st.expander("Technical Details", expanded=False):
        st.markdown("""
        <div style="font-size:0.82rem; color:#64748B; margin-bottom:0.5rem;">
        Raw model factors and contribution estimates for analyst review.
        </div>
        """, unsafe_allow_html=True)
        # Extract raw factors from explanation_summary if possible
        raw_factors = {}
        for fname in ["demand", "inventory", "competitor", "margin", "cost"]:
            col_name = f"explanation_{fname}_direction"
            if col_name in row:
                raw_factors[f"{fname}_direction"] = str(row[col_name])
        # Add other raw fields
        for col in ["demand_trend", "price_elasticity", "sales_volume", "inventory_level", "competitor_price", "cost_price", "profit_margin"]:
            if col in row:
                try:
                    raw_factors[col] = float(row[col])
                except (ValueError, TypeError):
                    raw_factors[col] = str(row[col])
        st.json(raw_factors)

        if explanation_text_raw and len(explanation_text_raw) > 5:
            st.markdown("**Full Explanation Text:**")
            st.code(explanation_text_raw, language="text")


# ═══════════════════════════════════════════════════════════════════════════
# REPORTS TAB
# ═══════════════════════════════════════════════════════════════════════════

def render_reports() -> None:
    page_hdr("Reports", "Generate CSV and PDF reports.")
    df = require_data(prefer_analyzed=True)
    if df is None: return
    ins = st.session_state.insights or build_insights(df)
    report_pdf_path = st.session_state.get("re_pdf")
    report_text = st.session_state.get("summary_report")

    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", data=df_to_csv(df), file_name="pricing_analysis.csv", mime="text/csv", use_container_width=True)
    if not REPORTING_AVAILABLE:
        st.warning(f"PDF unavailable: {REPORTING_IMPORT_ERROR}")
        st.info("Install openpyxl and reportlab.")
        if "category" in df.columns:
            show_table(category_preview(df))
        return
    gen = ReportGenerator(st.session_state.config)
    with c2:
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Generating..."):
                st.session_state["re_pdf"] = gen.generate_pdf_report(df, ins)
                st.session_state["summary_report"] = build_summary_report_text(df, ins)
                report_pdf_path = st.session_state["re_pdf"]
                report_text = st.session_state["summary_report"]

        if report_pdf_path:
            with open(report_pdf_path, "rb") as f:
                st.download_button("Download PDF", f.read(), Path(report_pdf_path).name, mime="application/pdf", use_container_width=True)

        if st.button("Send Summary Report to Email", use_container_width=True):
            if not report_text:
                st.warning("Please generate a Summary Report first.")
            else:
                cfg = _get_notification_config()
                if not cfg["configured"] or not cfg["smtp_server"] or not cfg["smtp_port"] or not cfg["sender"] or not cfg["recipient"] or not cfg["smtp_password"]:
                    st.warning("SMTP configuration missing in .env.")
                else:
                    attachments = []
                    if report_pdf_path and Path(report_pdf_path).exists():
                        with open(report_pdf_path, "rb") as f:
                            attachments.append((Path(report_pdf_path).name, f.read(), "application/pdf"))
                    success, status = send_email_alert(
                        "AI Pricing Platform — Summary Report",
                        report_text,
                        cfg["recipient"],
                        cfg["sender"],
                        cfg["smtp_server"],
                        cfg["smtp_port"],
                        cfg["smtp_password"],
                        attachments=attachments,
                    )
                    if success:
                        st.success("Summary report emailed successfully.")
                    else:
                        st.error("Unable to send summary report email. Check SMTP configuration.")

    if "category" in df.columns:
        st.subheader("Preview")
        show_table(category_preview(df))


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


def render_notifications_alerts() -> None:
    page_hdr("Notifications & Alerts", "Configure notification rules, email delivery, and notification history.")

    st.markdown(
        "<div style='background:#FFFFFF; border:1px solid #E5E7EB; border-radius:14px; padding:1.25rem; box-shadow:0 2px 10px rgba(15,23,42,0.06); margin-bottom:1rem;'>"
        "<p style='margin:0 0 0.75rem; color:#475569;'>Manage email alert preferences, frequency, and recipient settings for professional pricing notifications.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    env_loaded = _get_notification_config().get("configured", False)

    st.markdown("## Alert Settings")
    st.markdown("#### Custom Price Mismatch Alerts")
    send_validation_email = st.checkbox(
        "Send Custom Price Mismatch via Email",
        key="price_validation_send_email",
    )
    st.caption("Receive a concise email summary whenever price refresh validation detects pricing mismatches.")
    if send_validation_email and not env_loaded:
        st.warning("SMTP configuration missing in .env. Email notification skipped.")

    st.markdown("## Price Refresh Validation")
    with st.expander("Upload Flat File", expanded=True):
        st.caption("Expected columns: product_id (optional), product_name, custom_price")
        refresh_file = st.file_uploader(
            "Upload Refreshed Price File",
            type=["csv", "xlsx"],
            key="price_refresh_validation_upload",
        )
        compare_col, status_col = st.columns([0.35, 0.65])
        with compare_col:
            compare_clicked = st.button("Compare Prices", type="primary", use_container_width=True)
        with status_col:
            source_name = st.session_state.uploaded_name or "No dataset loaded"
            st.caption(f"Analysis dataset: {source_name}")

        if compare_clicked:
            analysis_df = get_df(prefer_analyzed=True)
            if analysis_df is None or analysis_df.empty:
                st.warning("Upload and process the main analysis dataset before running price validation.")
            else:
                refreshed_df, load_error = _load_price_refresh_file(refresh_file)
                if load_error:
                    st.warning(load_error)
                else:
                    results_df, summary, warnings = compare_price_files(analysis_df, refreshed_df)
                    for warning in warnings:
                        st.warning(warning)
                    if results_df.empty:
                        st.warning("No valid price comparison results were generated.")
                    else:
                        st.session_state.price_validation_results = results_df
                        st.session_state.price_validation_summary = summary
                        mismatch_count = summary.get("price_changed_count", 0)
                        if mismatch_count:
                            st.success(f"✓ {mismatch_count:,} price changes detected.")
                        else:
                            st.success("✓ No pricing mismatches found.")

                        if send_validation_email and mismatch_count:
                            success, status = send_price_validation_email(
                                results_df,
                                summary,
                                st.session_state.uploaded_name or "Loaded dataset",
                            )
                            if success:
                                st.success("✓ Summary email sent successfully.")
                            elif "SMTP configuration missing" in status:
                                st.warning("⚠ SMTP configuration missing in .env.\nComparison completed without email notification.")
                            else:
                                st.error(f"Email notification failed: {status}")

    results_df = st.session_state.price_validation_results
    summary = st.session_state.price_validation_summary or {}
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        st.markdown("#### Comparison Summary")
        metric_cols = st.columns(5)
        safe_metric(metric_cols[0], "Total Products Compared", f"{summary.get('total_compared', 0):,}")
        safe_metric(metric_cols[1], "Matching Prices", f"{summary.get('matching_prices', 0):,}")
        safe_metric(metric_cols[2], "Price Changes Detected", f"{summary.get('price_changed_count', 0):,}")
        safe_metric(metric_cols[3], "Missing Products", f"{summary.get('missing_products', 0):,}")
        safe_metric(metric_cols[4], "New Products", f"{summary.get('new_products', 0):,}")

        st.markdown("#### Comparison Results")
        display_df = results_df.copy()
        status_colors = {
            "NO CHANGE": "background-color: #DCFCE7; color: #166534; font-weight: 700;",
            "PRICE CHANGED": "background-color: #FFEDD5; color: #9A3412; font-weight: 700;",
            "MISSING IN REFRESH FILE": "background-color: #FEE2E2; color: #991B1B; font-weight: 700;",
            "NEW PRODUCT": "background-color: #FEE2E2; color: #991B1B; font-weight: 700;",
        }

        def _style_price_status(value: Any) -> str:
            return status_colors.get(str(value), "")

        styled_df = display_df.style.format({
            "current_price": _format_price_value,
            "custom_price": _format_price_value,
            "difference": _format_difference_value,
        })
        try:
            styled_df = styled_df.map(_style_price_status, subset=["status"])
        except AttributeError:
            styled_df = styled_df.applymap(_style_price_status, subset=["status"])
        try:
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=min(620, max(260, 38 * (len(display_df) + 1))))
        except Exception:
            show_table(display_df)

        mismatch_df = detect_price_mismatches(results_df)
        if not mismatch_df.empty:
            st.download_button(
                "Download Validation Results CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="price_refresh_validation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("## Notification History")
    history = st.session_state.notification_history or []
    if not history:
        st.info("No notification activity has been recorded yet.")
    else:
        history_df = pd.DataFrame(history)
        if not history_df.empty:
            show_table(history_df.head(12))
        else:
            st.info("No notification activity has been recorded yet.")


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
        "Notifications & Alerts": render_notifications_alerts,
        "Reports": render_reports,
        "Chat Assistant": render_chat_assistant,
    }
    dispatch.get(route, lambda: None)()

if __name__ == "__main__":
    main()
