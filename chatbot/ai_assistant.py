"""
AI Business Analyst Assistant

An intelligent chatbot that connects OpenAI's LLM capabilities to the
pricing platform's data analysis outputs. Users ask natural language
business questions and receive concise, data-driven answers.

Key Queries:
  - "Which products need repricing?"
  - "Why did profit decline?"
  - "Show risky products."
  - "What's our best-selling category?"
  - "How are we positioned against competitors?"

Architecture:
  User Query → Build Context → OpenAI API (with fallback) → Formatted Response

Features:
  - OpenAI-powered natural language understanding (gpt-4o-mini)
  - Structured system prompt with live DataFrame context
  - Conversation memory (last 10 turns preserved)
  - Rule-based fallback for all common queries (no API key required)
  - Data query functions for business insights extraction
  - Concise, business-focused formatting
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

from utils.config import AppConfig
from utils.helpers import format_currency, safe_divide

logger = logging.getLogger("ai_pricing.chatbot")


# ═══════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER - Improved Natural Language Understanding
# ═══════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Advanced intent classifier with keyword matching, fuzzy matching, and synonym support.
    
    Handles:
    - Exact keyword matching
    - Fuzzy string matching (partial/misspelled queries)
    - Multiple synonym support per intent
    - Semantic intent grouping
    - Order matters: More specific intents checked before general ones
    """

    # Define intents with comprehensive keyword/synonym mappings
    # IMPORTANT: More specific intents MUST come before their parent/general intents
    INTENTS = {
        # Specific pricing intents FIRST (before generic PRICING)
        "PRICING_INCREASE": {
            "keywords": [
                "where should we increase", "which products need price increase",
                "show products to raise", "what should be repriced upward",
                "which items are underpriced", "price increase recommendation",
                "raise price", "products to increase", "increase price",
                "need price increase", "price up", "price hike"
            ],
            "handler": "_handle_pricing_increase"
        },
        "PRICING_DECREASE": {
            "keywords": [
                "where should we decrease", "which products need markdown",
                "show overpriced", "price reduction", "price decrease",
                "which items should be cheaper", "markdown opportunity",
                "lower price", "products to decrease", "decrease price",
                "price down", "markdown"
            ],
            "handler": "_handle_pricing_decrease"
        },
        # Then general pricing (catches everything else pricing-related)
        "PRICING": {
            "keywords": [
                "repric", "re-price", "price change", "should i change",
                "need to change", "pricing recommendation", "repricing",
                "price adjustment", "optimal price", "what should be repriced",
                "which products need price", "price"
            ],
            "handler": "_handle_pricing"
        },
        # Specific top products BEFORE forecasting (which contains "popular", "selling")
        "TOP_PRODUCTS": {
            "keywords": [
                "best sell", "top product", "best perform", "top revenue",
                "highest revenue", "top earner", "best revenue", "top earning",
                "revenue leader", "best product"
            ],
            "handler": "_handle_top_products"
        },
        # Then forecasting
        "FORECASTING": {
            "keywords": [
                "demand", "forecast", "trend", "demand forecast",
                "future demand", "demand trend", "forecast outlook",
                "expected demand", "sales forecast", "predict", "prediction"
            ],
            "handler": "_handle_forecasting"
        },
        # Other intents
        "MARGINS": {
            "keywords": [
                "margin", "profit", "improve profit", "best margin", "low margin",
                "margin opportun", "profit opportun", "profit optim", "margin gain",
                "margin improvement", "highest profit", "profitability",
                "profit potential", "margin analysis", "which products can improve profit"
            ],
            "handler": "_handle_margins"
        },
        "RISK": {
            "keywords": [
                "risk", "risky", "danger", "flag", "critical", "high risk",
                "risk alert", "risky product", "risk summary", "risk assessment",
                "products needing attention", "highest risk", "risk item",
                "problem product", "concern", "alert"
            ],
            "handler": "_handle_risk"
        },
        "INVENTORY": {
            "keywords": [
                "inventory", "stock", "supply", "warehouse", "reorder", "overstock",
                "out of stock", "low stock", "excess inventory", "stock status",
                "stock risk", "inventory optim", "stock problem", "inventory action",
                "excess stock", "stock level"
            ],
            "handler": "_handle_inventory"
        },
        "COMPETITORS": {
            "keywords": [
                "competit", "market position", "how are we", "competitor", "against",
                "competitor pricing", "market comparison", "pricing gap", "competitive",
                "competitive position", "competitor intelligence", "competitive analysis"
            ],
            "handler": "_handle_competitors"
        },
        "PORTFOLIO": {
            "keywords": [
                "overview", "summary", "kpi", "dashboard", "how are we doing",
                "portfolio", "portfolio summary", "executive summary", "key insights",
                "where are we", "portfolio status"
            ],
            "handler": "_handle_portfolio"
        },
        "ALERTS": {
            "keywords": [
                "alert", "notif", "warning", "what's wrong", "issue", "problem",
                "active alert", "critical"
            ],
            "handler": "_handle_alerts"
        },
        "HELP": {
            "keywords": [
                "help", "what can you", "capabilit", "what do you know", "ask",
                "what questions", "what can i ask", "tell me"
            ],
            "handler": "_handle_help"
        },
        "GREETING": {
            "keywords": [
                "hi", "hello", "hey", "good morning", "good evening", "good day",
                "greet", "how are you", "howdy", "what's up"
            ],
            "handler": "_handle_greeting"
        },
    }

    @classmethod
    def classify(cls, query: str) -> Tuple[str, float]:
        """
        Classify a query into an intent category.

        Strategy:
        1. Try exact keyword match (highest confidence)
        2. Try fuzzy keyword match (partial matches)
        3. Return GENERAL if no match

        Args:
            query: User query string

        Returns:
            Tuple of (intent_name, confidence_score)
        """
        query_lower = query.lower()
        best_intent = "GENERAL"
        best_score = 0.0

        # Check intents in order (order matters - specific before general)
        for intent_name, intent_data in cls.INTENTS.items():
            keywords = intent_data["keywords"]

            # Try exact match (score: 1.0) - for this intent only
            for kw in keywords:
                if kw in query_lower:
                    return (intent_name, 1.0)

        # If no exact match found, try fuzzy match on all intents
        for intent_name, intent_data in cls.INTENTS.items():
            keywords = intent_data["keywords"]
            for kw in keywords:
                ratio = SequenceMatcher(None, query_lower, kw).ratio()
                if ratio > 0.75 and ratio > best_score:
                    best_intent = intent_name
                    best_score = ratio

        if best_score > 0.75:
            return (best_intent, best_score)

        return (best_intent, best_score)


# ─── Constants ─────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS: int = 10
MAX_CONTEXT_PRODUCTS: int = 50

SYSTEM_PROMPT_TEMPLATE: str = """You are an AI Business Analyst for the AI Pricing Intelligence Platform. You help business users understand their pricing data, identify opportunities, and make decisions.

PORTFOLIO OVERVIEW:
{portfolio_overview}

TOP PRODUCTS:
{top_products}

RISK SUMMARY:
{risk_summary}

COMPETITIVE POSITION:
{competitive_position}

INVENTORY STATUS:
{inventory_status}

PRICING RECOMMENDATIONS:
{pricing_recs}

ALERTS:
{alerts}

RULES:
1. Be concise and business-focused. Use bullet points where helpful.
2. Always reference specific data (product names, numbers, percentages).
3. If data is unavailable, say so — do not make up information.
4. For "which products need repricing" — list products with "Increase"/"Decrease" recommendations.
5. For "why did profit decline" — explain margin, volume, and cost factors.
6. For "show risky products" — list high-risk products with scores and primary factors.
7. Use natural business language. Say "risk score" not "composite_risk_score".
8. Format currency as $X,XXX.XX and percentages as X.X%.
9. Keep responses under 200 words unless details are requested.
10. Greet briefly and naturally."""


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role: str = "user"
    content: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


# ═══════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _fmtd(v: float) -> str:
    """Format a number as currency: $1,234.56"""
    try:
        if abs(v) >= 1000000:
            return f"${v/1000000:,.2f}M"
        elif abs(v) >= 1000:
            return f"${v:,.2f}"
        else:
            return f"${v:.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _fmtp(v: float) -> str:
    """Format a number as percentage: +8.5% or -3.2%"""
    try:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _fmt_count(count: int, total: int) -> str:
    """Format a count summary like 'Showing 10 of 74 products'."""
    if count >= total:
        return f"All {total} products"
    return f"Showing **{count}** of **{total}** products"


def _section_header(title: str, count: int = 0, total: int = 0) -> str:
    """Create a bold section header with optional product count."""
    if count > 0 and total > 0:
        summary = _fmt_count(count, total)
        return f"#### **{title}** ({summary})"
    elif count > 0:
        return f"#### **{title}** — {count} products"
    return f"#### **{title}**"


def _product_bullet(name: str, details: List[str]) -> str:
    """Create a formatted product bullet point."""
    detail_str = " • ".join(details)
    return f"\n**{name}**\n\n{detail_str}"


def _separator() -> str:
    return "\n\n---\n\n"


def _subsection(title: str) -> str:
    return f"\n**{title}**\n"


def _normalise_assistant_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare analysis outputs for safe read-only chatbot access."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    numeric_cols = [
        "current_price", "cost_price", "competitor_price", "sales_volume",
        "inventory_level", "revenue", "expected_revenue", "expected_profit",
        "profit_margin", "margin_percentage", "price_change_pct",
        "composite_risk_score", "competitive_score", "price_gap_pct",
        "days_of_cover", "predicted_demand", "forecast_next_30d",
        "forecast_confidence", "demand_trend",
    ]
    numeric_cols.extend([c for c in out.columns if c.startswith("forecast_week_")])
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in ["product_name", "category", "recommendation", "risk_level", "stock_status", "inventory_action", "market_position", "demand_trend_category"]:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)
    return out.replace([np.inf, -np.inf], 0)


def _clean_response_start(text: str) -> str:
    """Keep responses readable in Streamlit and Windows consoles."""
    if not text:
        return ""
    cleaned = text.strip()
    while cleaned and not cleaned[0].isascii():
        cleaned = cleaned[1:].lstrip()
    while cleaned and cleaned[0] in "#* _-:|>":
        cleaned = cleaned[1:].lstrip()
        while cleaned and not cleaned[0].isascii():
            cleaned = cleaned[1:].lstrip()
    lines = cleaned.splitlines()
    if lines:
        lines[0] = lines[0].replace("*", "").strip()
        cleaned = "\n".join(lines)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDERS  (extract business summaries from DataFrame)
# ═══════════════════════════════════════════════════════════════════════════

def _build_portfolio_overview(df: pd.DataFrame, insights: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"- Total products: {len(df)}")
    if "category" in df.columns:
        lines.append(f"- Categories: {df['category'].nunique()}")
    for col in ["expected_revenue", "revenue"]:
        if col in df.columns:
            lines.append(f"- Total revenue: ${df[col].sum():,.2f}")
            break
    if not any("revenue" in l for l in lines):
        if "current_price" in df.columns and "sales_volume" in df.columns:
            lines.append(f"- Estimated revenue: ${(df['current_price'] * df['sales_volume']).sum():,.2f}")
    if "expected_profit" in df.columns:
        lines.append(f"- Total profit: ${df['expected_profit'].sum():,.2f}")
    for col in ["margin_percentage", "profit_margin"]:
        if col in df.columns:
            if col == "profit_margin":
                lines.append(f"- Average margin: {df[col].mean() * 100:.1f}%")
            else:
                lines.append(f"- Average margin: {df[col].mean():.1f}%")
            break
    if "composite_risk_score" in df.columns:
        lines.append(f"- Average risk score: {df['composite_risk_score'].mean():.1f}/100")
        lines.append(f"- High-risk products (≥70): {(df['composite_risk_score'] >= 70).sum()}")
    if "recommendation" in df.columns:
        for rec in ["Increase", "Decrease", "Maintain"]:
            c = int((df["recommendation"] == rec).sum())
            if c > 0:
                lines.append(f"- Recommend {rec.lower()}: {c}")
    return "\n".join(lines)


def _build_top_products(df: pd.DataFrame) -> str:
    lines: List[str] = []
    sort_col = "expected_revenue" if "expected_revenue" in df.columns else "revenue" if "revenue" in df.columns else None
    if sort_col is None and "current_price" in df.columns and "sales_volume" in df.columns:
        df = df.copy()
        df["_rev"] = df["current_price"] * df["sales_volume"]
        sort_col = "_rev"
    if sort_col is not None:
        col = sort_col
        top_df = df.nlargest(min(MAX_CONTEXT_PRODUCTS, len(df)), col)
        for _, row in top_df.iterrows():
            name = str(row.get("product_name", ""))[:25]
            cat = str(row.get("category", ""))
            price = row.get("current_price", 0)
            margin = None
            for c in ["margin_percentage", "profit_margin"]:
                if c in row:
                    margin = row[c]
                    break
            risk = row.get("composite_risk_score", None)
            rec = row.get("recommendation", "")
            parts = [f"{name} ({cat})", f"${price:.2f}"]
            if margin is not None:
                parts.append(f"{margin*100:.1f}% margin" if margin < 1 else f"{margin:.1f}% margin")
            if risk is not None:
                parts.append(f"risk {risk:.0f}")
            if rec:
                parts.append(f"→ {rec}")
            lines.append(f"- {' | '.join(parts)}")
    return "\n".join(lines)


def _build_risk_summary(df: pd.DataFrame) -> str:
    lines: List[str] = []
    level_col = "risk_level" if "risk_level" in df.columns else "risk_category" if "risk_category" in df.columns else None
    score_col = "composite_risk_score"
    if level_col and level_col in df.columns:
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            c = int((df[level_col] == level).sum())
            if c > 0:
                lines.append(f"- {level.title()}: {c}")
    if score_col in df.columns:
        top = df.nlargest(3, score_col)
        if not top.empty:
            lines.append("")
            lines.append("Highest risk products:")
            for _, row in top.iterrows():
                name = row.get("product_name", "Unknown")
                score = row.get(score_col, 0)
                factor = row.get("primary_risk_factor", "")
                lines.append(f"  - {name}: {score:.0f}/100 ({factor})")
    return "\n".join(lines)


def _build_competitive_position(df: pd.DataFrame) -> str:
    lines: List[str] = []
    if "market_position" in df.columns:
        for pos in ["Premium", "Competitive", "Discount", "Aggressive Discount"]:
            c = int((df["market_position"] == pos).sum())
            if c > 0:
                lines.append(f"- {pos}: {c} ({(c/len(df)*100):.0f}%)")
    for flag, label in [("competitor_undercut_flag", "being undercut"), ("overpriced_flag", "overpriced"), ("underpriced_flag", "underpriced")]:
        if flag in df.columns:
            c = int(df[flag].sum())
            if c > 0:
                lines.append(f"- {c} products {label}")
    return "\n".join(lines)


def _build_inventory_status(df: pd.DataFrame) -> str:
    lines: List[str] = []
    if "stock_status" in df.columns:
        for s in ["Low Stock", "Out of Stock", "Overstocked"]:
            c = int((df["stock_status"] == s).sum())
            if c > 0:
                lines.append(f"- {s}: {c}")
    if "inventory_action" in df.columns:
        for a in ["increase", "discount"]:
            c = int((df["inventory_action"] == a).sum())
            if c > 0:
                lines.append(f"- Recommend {a} price for {c} products")
    if "days_of_cover" in df.columns:
        lines.append(f"- Avg days of cover: {df['days_of_cover'].mean():.0f}")
    return "\n".join(lines)


def _build_pricing_recs(df: pd.DataFrame) -> str:
    lines: List[str] = []
    if "recommendation" in df.columns:
        for rec in ["Increase", "Decrease", "Maintain"]:
            c = int((df["recommendation"] == rec).sum())
            if c > 0:
                lines.append(f"- {rec}: {c}")
        inc = df[df["recommendation"] == "Increase"]
        if not inc.empty:
            lines.append("")
            lines.append("Increase candidates:")
            for _, row in inc.nlargest(3, "price_change_pct" if "price_change_pct" in inc.columns else "current_price").iterrows():
                n = row.get("product_name", "")
                p = row.get("price_change_pct", 0)
                c = row.get("current_price", 0)
                o = row.get("optimal_price", 0)
                lines.append(f"  - {n}: +{p:.0f}% (${c:.2f}→${o:.2f})")
        dec = df[df["recommendation"] == "Decrease"]
        if not dec.empty:
            lines.append("")
            lines.append("Decrease candidates:")
            for _, row in dec.nlargest(3, "price_change_pct" if "price_change_pct" in dec.columns else "current_price").iterrows():
                n = row.get("product_name", "")
                p = abs(row.get("price_change_pct", 0))
                c = row.get("current_price", 0)
                o = row.get("optimal_price", 0)
                lines.append(f"  - {n}: -{p:.0f}% (${c:.2f}→${o:.2f})")
    return "\n".join(lines)


def _build_alerts_summary(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return "No active alerts."
    lines: List[str] = []
    for a in alerts[:5]:
        sev = a.get("severity", "Info")
        cat = a.get("category", "")
        msg = str(a.get("message", ""))[:100]
        lines.append(f"- [{sev}] {cat}: {msg}")
    if len(alerts) > 5:
        lines.append(f"- +{len(alerts)-5} more alerts")
    return "\n".join(lines)


def _build_system_context(df: pd.DataFrame, insights: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        portfolio_overview=_build_portfolio_overview(df, insights),
        top_products=_build_top_products(df),
        risk_summary=_build_risk_summary(df),
        competitive_position=_build_competitive_position(df),
        inventory_status=_build_inventory_status(df),
        pricing_recs=_build_pricing_recs(df),
        alerts=_build_alerts_summary(alerts),
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPENAI CALL
# ═══════════════════════════════════════════════════════════════════════════

def _call_openai(messages: List[Dict[str, str]], api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini", temperature: float = 0.3,
                 max_tokens: int = 600) -> Optional[str]:
    """Call the OpenAI chat completions API. Returns response text or None."""
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not installed")
        return None
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("No OpenAI API key found")
        return None
    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED FALLBACK RESPONSE WITH INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _fallback_response(query: str, df: pd.DataFrame,
                       insights: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
    """
    Generate rule-based responses for business questions using intent classification.
    
    Strategy:
    1. Classify query intent using IntentClassifier
    2. Route to appropriate intent handler
    3. Fallback to generic help if intent unclear
    """
    safe_df = _normalise_assistant_df(df)
    if safe_df.empty:
        return "No data is available yet. Please load a dataset and run the analysis pipeline first."

    query_lower = query.lower()
    intent_name, confidence = IntentClassifier.classify(query_lower)

    try:
        if intent_name == "GREETING":
            return _handle_greeting(query)
        elif intent_name == "PRICING_INCREASE":
            return _handle_pricing_increase(safe_df)
        elif intent_name == "PRICING_DECREASE":
            return _handle_pricing_decrease(safe_df)
        elif intent_name == "PRICING":
            return _handle_pricing(safe_df)
        elif intent_name == "MARGINS":
            return _handle_margins(safe_df)
        elif intent_name == "RISK":
            return _handle_risk(safe_df)
        elif intent_name == "INVENTORY":
            return _handle_inventory(safe_df)
        elif intent_name == "FORECASTING":
            return _handle_forecasting(safe_df)
        elif intent_name == "COMPETITORS":
            return _handle_competitors(safe_df)
        elif intent_name == "PORTFOLIO":
            return _handle_portfolio(safe_df, insights)
        elif intent_name == "ALERTS":
            return _handle_alerts(alerts)
        elif intent_name == "TOP_PRODUCTS":
            return _handle_top_products(safe_df)
        elif intent_name == "HELP":
            return _handle_help()
        else:
            return _handle_portfolio(safe_df, insights)
    except Exception as exc:
        logger.exception("Rule-based assistant response failed")
        return (
            "I found the analysis context, but that specific view could not be rendered safely. "
            f"Try a pricing, risk, margin, inventory, forecast, competitor, or portfolio question. Details: {exc}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INTENT HANDLERS — Professional Markdown Formatting
# ═══════════════════════════════════════════════════════════════════════════

def _handle_greeting(query: str) -> str:
    """Handle greeting queries."""
    return (
        "Hello! I'm your **AI Business Analyst**. I can help you with:\n\n"
        "• **Pricing recommendations** and repricing analysis\n"
        "• **Margin and profit** opportunities\n"
        "• **Risk assessment** and product flagging\n"
        "• **Inventory optimization**\n"
        "• **Demand forecasting**\n"
        "• **Competitive intelligence**\n"
        "• **Portfolio overview** and alerts\n\n"
        "Ask me anything about your pricing data."
    )


def _handle_pricing(df: pd.DataFrame) -> str:
    """Handle general pricing repricing queries."""
    if "recommendation" not in df.columns:
        return "⚠️ Please run the **pricing engine** first to get repricing recommendations."

    inc = df[df["recommendation"] == "Increase"]
    dec = df[df["recommendation"] == "Decrease"]
    maintain = df[df["recommendation"] == "Maintain"]

    if inc.empty and dec.empty:
        return "✅ All products are **optimally priced**. No repricing needed at this time."

    lines = []
    lines.append(_section_header("📊 **PRICING RECOMMENDATIONS**"))
    lines.append("")
    lines.append("")

    total_recs = len(inc) + len(dec)

    if not inc.empty:
        lines.append(_subsection(f"🔺 **PRICE INCREASE — {len(inc)} products**"))
        show_count = min(10, len(inc))
        for _, r in inc.head(show_count).iterrows():
            name = str(r.get("product_name", "Unknown"))[:50]
            pct = r.get("price_change_pct", 0)
            new_price = r.get("optimal_price", 0)
            current_price = r.get("current_price", 0)
            lines.append(f"\n🔹 **{name}**")
            lines.append(f"   • Current Price: {_fmtd(current_price)}")
            lines.append(f"   • Recommended: {_fmtd(new_price)} ( **+{pct:.0f}%** )")
        if len(inc) > show_count:
            remaining = len(inc) - show_count
            lines.append(f"\n   *...and {remaining} more products with price increase recommendations.*")
        lines.append("")

    if not dec.empty:
        if not inc.empty:
            lines.append(_separator())
        lines.append(_subsection(f"🔻 **PRICE DECREASE — {len(dec)} products**"))
        show_count = min(10, len(dec))
        for _, r in dec.head(show_count).iterrows():
            name = str(r.get("product_name", "Unknown"))[:50]
            pct = abs(r.get("price_change_pct", 0))
            new_price = r.get("optimal_price", 0)
            current_price = r.get("current_price", 0)
            lines.append(f"\n🔹 **{name}**")
            lines.append(f"   • Current Price: {_fmtd(current_price)}")
            lines.append(f"   • Recommended: {_fmtd(new_price)} ( **-{pct:.0f}%** )")
        if len(dec) > show_count:
            remaining = len(dec) - show_count
            lines.append(f"\n   *...and {remaining} more products with price decrease recommendations.*")
        lines.append("")

    if not maintain.empty:
        lines.append(f"\n✅ **Price Maintain (optimal): {len(maintain)} products**")

    lines.append(f"\n\n*{_fmt_count(total_recs, total_recs)} with pricing recommendations.*")

    return "\n".join(lines)


def _handle_pricing_increase(df: pd.DataFrame) -> str:
    """Handle price increase queries."""
    if "recommendation" not in df.columns:
        return "⚠️ Please run the **pricing engine** first to get repricing recommendations."

    inc = df[df["recommendation"] == "Increase"]
    if inc.empty:
        return "✅ **No price increase opportunities identified** at this time. All products are optimally priced or need decreases."

    total = len(inc)
    show_count = min(10, total)
    lines = []
    lines.append(_section_header("🔺 **PRICE INCREASE OPPORTUNITIES**", total, total))
    lines.append("")
    lines.append("")

    for _, r in inc.head(show_count).iterrows():
        name = str(r.get("product_name", "Unknown"))[:50]
        pct = r.get("price_change_pct", 0)
        new_price = r.get("optimal_price", 0)
        current_price = r.get("current_price", 0)
        lines.append(f"🔹 **{name}**")
        lines.append(f"   • **Current Price:** {_fmtd(current_price)}")
        lines.append(f"   • **Recommended Price:** {_fmtd(new_price)}")
        lines.append(f"   • **Suggested Change:** +{pct:.0f}%")
        lines.append("")

    if total > show_count:
        remaining = total - show_count
        lines.append(f"*Showing {show_count} of {total} products. {remaining} more products with increase recommendations.*\n")

    lines.append(f"*Recommendation: Consider raising prices on these products to capture additional revenue.*")

    return "\n".join(lines)


def _handle_pricing_decrease(df: pd.DataFrame) -> str:
    """Handle price decrease queries."""
    if "recommendation" not in df.columns:
        return "⚠️ Please run the **pricing engine** first to get repricing recommendations."

    dec = df[df["recommendation"] == "Decrease"]
    if dec.empty:
        return "✅ **No price decrease opportunities identified** at this time. All products are optimally priced."

    total = len(dec)
    show_count = min(10, total)
    lines = []
    lines.append(_section_header("🔻 **PRICE DECREASE OPPORTUNITIES**", total, total))
    lines.append("")
    lines.append("")

    for _, r in dec.head(show_count).iterrows():
        name = str(r.get("product_name", "Unknown"))[:50]
        pct = abs(r.get("price_change_pct", 0))
        new_price = r.get("optimal_price", 0)
        current_price = r.get("current_price", 0)
        lines.append(f"🔹 **{name}**")
        lines.append(f"   • **Current Price:** {_fmtd(current_price)}")
        lines.append(f"   • **Recommended Price:** {_fmtd(new_price)}")
        lines.append(f"   • **Suggested Change:** -{pct:.0f}%")
        lines.append("")

    if total > show_count:
        remaining = total - show_count
        lines.append(f"*Showing {show_count} of {total} products. {remaining} more products with decrease recommendations.*\n")

    lines.append(f"*Recommendation: Consider reducing prices to improve competitiveness and demand.*")

    return "\n".join(lines)


def _handle_margins(df: pd.DataFrame) -> str:
    """Handle margin and profit opportunity queries."""
    lines = []
    lines.append(_section_header("💰 **MARGIN & PROFIT ANALYSIS**"))
    lines.append("")
    lines.append("")

    margin_col = None
    for col in ["margin_percentage", "profit_margin"]:
        if col in df.columns:
            margin_col = col
            break

    if margin_col:
        avg_margin = df[margin_col].mean()
        if margin_col == "profit_margin":
            avg_margin *= 100
        lines.append(f"📊 **Average Margin:** {avg_margin:.1f}%")
        lines.append("")

    if "expected_profit" in df.columns:
        total_profit = df["expected_profit"].sum()
        lines.append(f"📊 **Total Projected Profit:** {_fmtd(total_profit)}")
        lines.append("")

    # Find low-margin products
    if margin_col:
        threshold = 0.10 if margin_col == "profit_margin" else 10
        low_margin = df[df[margin_col] < threshold]
        total_low = len(low_margin)
        if total_low > 0:
            lines.append(_subsection(f"⚠️ **Low-Margin Products: {total_low}**"))
            lines.append("")
            show_count = min(5, total_low)
            for _, r in low_margin.head(show_count).iterrows():
                name = str(r.get("product_name", "Unknown"))[:50]
                margin = r[margin_col] * 100 if margin_col == "profit_margin" else r[margin_col]
                lines.append(f"   • **{name}:** {margin:.1f}% margin")
            if total_low > show_count:
                lines.append(f"   *...and {total_low - show_count} more low-margin products*")
            lines.append("")

    # Find high-margin opportunities
    if margin_col:
        high_threshold = 0.40 if margin_col == "profit_margin" else 40
        high_margin = df[df[margin_col] >= high_threshold]
        if len(high_margin) > 0:
            lines.append(_subsection(f"✅ **Strong Margin Products: {len(high_margin)}**"))
            lines.append("")

    lines.append(f"\n📌 **Tip:** Navigate to the **Risk & Explainability** dashboard for detailed analysis.")

    return "\n".join(lines)


def _handle_risk(df: pd.DataFrame) -> str:
    """Handle risk product queries."""
    if "composite_risk_score" not in df.columns:
        return "⚠️ Run the **risk assessment** first to see risk data."

    high_risk = df[df["composite_risk_score"] >= 70]
    if high_risk.empty:
        return "✅ **Your portfolio is healthy.** No high-risk products identified."

    total = len(high_risk)
    show_count = min(10, total)
    lines = []
    lines.append(_section_header("🚨 **HIGH-RISK PRODUCTS**", total, total))
    lines.append("")
    lines.append("")

    for _, r in high_risk.head(show_count).iterrows():
        name = str(r.get("product_name", "Unknown"))[:50]
        score = r["composite_risk_score"]
        factor = r.get("primary_risk_factor", "Multiple factors")
        lines.append(f"⚠️ **{name}**")
        lines.append(f"   • **Risk Score:** {score:.0f}/100")
        lines.append(f"   • **Primary Issue:** {factor}")
        lines.append("")

    if total > show_count:
        remaining = total - show_count
        lines.append(f"*Showing {show_count} of {total} products. {remaining} more high-risk products.*\n")
        lines.append("")

    # Add risk level distribution
    level_col = "risk_level" if "risk_level" in df.columns else None
    if level_col and level_col in df.columns:
        lines.append(f"**Risk Distribution:**")
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            c = int((df[level_col] == level).sum())
            if c > 0:
                icon = "🔴" if level == "CRITICAL" else "🟠" if level == "HIGH" else "🟡" if level == "MEDIUM" else "🟢"
                lines.append(f"   {icon} {level.title()}: {c}")

    return "\n".join(lines)


def _handle_inventory(df: pd.DataFrame) -> str:
    """Handle inventory status queries."""
    lines = []
    lines.append(_section_header("📦 **INVENTORY STATUS**"))
    lines.append("")
    lines.append("")

    if "stock_status" in df.columns:
        statuses = []
        for status in ["Low Stock", "Out of Stock", "Overstocked", "Healthy"]:
            count = int((df["stock_status"] == status).sum())
            if count > 0:
                statuses.append(f"{status}: {count}")

        if statuses:
            lines.append("**Stock Status Distribution:**")
            for s in statuses:
                icon = "🟢" if "Healthy" in s else "🔴" if "Out" in s else "🟠" if "Low" in s else "🔵"
                lines.append(f"   {icon} {s}")
            lines.append("")

    if "days_of_cover" in df.columns:
        avg_doc = df["days_of_cover"].mean()
        lines.append(f"📊 **Average Days of Cover:** {avg_doc:.0f} days")
        lines.append("")

    if "inventory_action" in df.columns:
        increase = int((df["inventory_action"] == "increase").sum())
        discount = int((df["inventory_action"] == "discount").sum())

        if increase > 0 or discount > 0:
            lines.append("**Recommended Actions:**")
            if increase > 0:
                lines.append(f"   🔺 **{increase} products:** Increase price (low stock)")
            if discount > 0:
                lines.append(f"   🔻 **{discount} products:** Discount price (overstock)")
            lines.append("")

    # Show critical inventory items
    if "stock_status" in df.columns:
        critical_stock = df[df["stock_status"].isin(["Low Stock", "Out of Stock"])]
        if not critical_stock.empty:
            total = len(critical_stock)
            show_count = min(5, total)
            lines.append(_subsection(f"⚠️ **Stock Action Needed: {total} products**"))
            lines.append("")
            for _, r in critical_stock.head(show_count).iterrows():
                name = str(r.get("product_name", "Unknown"))[:50]
                status = r.get("stock_status", "Unknown")
                level = int(r.get("inventory_level", 0))
                lines.append(f"   • **{name}** — {status} (Level: {level:,})")
            if total > show_count:
                lines.append(f"   *...and {total - show_count} more products needing attention*")

    return "\n".join(lines)


def _handle_forecasting(df: pd.DataFrame) -> str:
    """Handle demand forecasting queries."""
    lines = []
    lines.append(_section_header("📈 **DEMAND FORECAST**"))
    lines.append("")
    lines.append("")

    if "demand_trend_category" in df.columns:
        increasing = int((df["demand_trend_category"] == "increasing").sum())
        stable = int((df["demand_trend_category"] == "stable").sum())
        declining = int((df["demand_trend_category"] == "declining").sum())

        lines.append("**Trend Distribution:**")
        if increasing > 0:
            lines.append(f"   📈 **Increasing:** {increasing} products")
        if stable > 0:
            lines.append(f"   ➡️ **Stable:** {stable} products")
        if declining > 0:
            lines.append(f"   📉 **Declining:** {declining} products")
        lines.append("")
    elif "demand_trend" in df.columns:
        avg_demand = df["demand_trend"].mean()
        lines.append(f"📊 **Average Demand Trend:** {avg_demand:.2f}/1.0")
        lines.append("")

    if "forecast_next_30d" in df.columns:
        forecast_30d = df["forecast_next_30d"].sum()
        lines.append(f"📊 **30-Day Forecast:** {forecast_30d:,.0f} units")
        lines.append("")

    if "predicted_demand" in df.columns:
        avg_pred = df["predicted_demand"].mean()
        lines.append(f"📊 **Average Predicted Demand:** {avg_pred:,.0f} units")
        lines.append("")

    if "forecast_confidence" in df.columns:
        avg_conf = df["forecast_confidence"].mean() * 100
        lines.append(f"🎯 **Average Forecast Confidence:** {avg_conf:.0f}%")
        lines.append("")

    # Top increasing products
    if "demand_trend_category" in df.columns:
        growing = df[df["demand_trend_category"] == "increasing"].nlargest(3, "predicted_demand" if "predicted_demand" in df.columns else "sales_volume")
        if not growing.empty:
            lines.append(_subsection("🔥 **Top Growing Products**"))
            lines.append("")
            for _, r in growing.iterrows():
                name = str(r.get("product_name", "Unknown"))[:50]
                demand = r.get("predicted_demand", r.get("sales_volume", 0))
                lines.append(f"   • **{name}** — Forecast: {demand:,.0f} units")

    return "\n".join(lines)


def _handle_competitors(df: pd.DataFrame) -> str:
    """Handle competitive positioning queries."""
    lines = []
    lines.append(_section_header("🏢 **COMPETITIVE POSITION**"))
    lines.append("")
    lines.append("")

    if "market_position" in df.columns:
        positions = ["Premium", "Competitive", "Discount", "Aggressive Discount"]
        lines.append("**Market Position Distribution:**")
        for pos in positions:
            count = int((df["market_position"] == pos).sum())
            if count > 0:
                pct = (count / len(df)) * 100
                icon = "🔴" if pos == "Premium" else "🟢" if pos == "Competitive" else "🟠" if pos == "Discount" else "🔵"
                lines.append(f"   {icon} **{pos}:** {count} products ({pct:.0f}%)")
        lines.append("")

    if "competitive_score" in df.columns:
        avg_score = df["competitive_score"].mean()
        lines.append(f"📊 **Average Competitive Score:** {avg_score:.1f}/100")
        lines.append("")

    # Flag issues
    issues = []
    for flag, label in [("overpriced_flag", "may be overpriced"),
                       ("underpriced_flag", "may be underpriced"),
                       ("competitor_undercut_flag", "being undercut by competitors")]:
        if flag in df.columns:
            count = int(df[flag].sum())
            if count > 0:
                issues.append(f"⚠️ {count} products {label}")

    if issues:
        lines.append("**Competitive Issues:**")
        for issue in issues:
            lines.append(f"   {issue}")
        lines.append("")

    # Top competitive threats
    if "competitor_undercut_flag" in df.columns and df["competitor_undercut_flag"].sum() > 0:
        undercut = df[df["competitor_undercut_flag"] == True].nlargest(3, "price_gap_pct" if "price_gap_pct" in df.columns else "current_price")
        if not undercut.empty:
            lines.append(_subsection("⚠️ **Top Competitor Threats**"))
            for _, r in undercut.iterrows():
                name = str(r.get("product_name", "Unknown"))[:50]
                gap = r.get("price_gap_pct", 0)
                lines.append(f"   • **{name}** — Gap: {gap:+.1f}%")

    return "\n".join(lines)


def _handle_portfolio(df: pd.DataFrame, insights: Dict[str, Any]) -> str:
    """Handle portfolio overview queries."""
    overview = _build_portfolio_overview(df, insights)
    lines = []
    lines.append(_section_header("📋 **PORTFOLIO OVERVIEW**"))
    lines.append("")
    lines.append(overview)

    # Add insights summary
    if insights:
        lines.append("")
        lines.append("**Key Metrics:**")
        if "current_revenue" in insights:
            lines.append(f"   • Revenue: {_fmtd(insights['current_revenue'])}")
        if "expected_revenue" in insights and insights["expected_revenue"] != insights.get("current_revenue", 0):
            lines.append(f"   • Expected Revenue: {_fmtd(insights['expected_revenue'])}")
        if "avg_margin" in insights:
            lines.append(f"   • Avg Margin: {insights['avg_margin']*100:.1f}%")
        if "avg_risk_score" in insights:
            lines.append(f"   • Avg Risk Score: {insights['avg_risk_score']:.1f}/100")
        if "high_risk_count" in insights:
            lines.append(f"   • High-Risk Products: {insights['high_risk_count']}")

    return "\n".join(lines)


def _handle_alerts(alerts: List[Dict[str, Any]]) -> str:
    """Handle alert queries."""
    if not alerts:
        return "✅ **No active alerts.** Your portfolio is in good shape."

    lines = []
    lines.append(_section_header("🔔 **ACTIVE ALERTS**", len(alerts)))
    lines.append("")
    lines.append("")

    critical = [a for a in alerts if a.get("severity") in ("Critical", "CRITICAL")]
    high = [a for a in alerts if a.get("severity") in ("High", "HIGH")]
    medium = [a for a in alerts if a.get("severity") in ("Medium", "MEDIUM", "Warning", "WARNING")]
    info = [a for a in alerts if a.get("severity") in ("Info", "INFO", "Low", "LOW")]

    if critical:
        lines.append(f"🔴 **CRITICAL: {len(critical)}**")
        for a in critical[:5]:
            msg = str(a.get("message", ""))[:80]
            lines.append(f"   • {msg}")
        if len(critical) > 5:
            lines.append(f"   *...and {len(critical) - 5} more*")
        lines.append("")

    if high:
        lines.append(f"🟠 **HIGH: {len(high)}**")
        for a in high[:5]:
            msg = str(a.get("message", ""))[:80]
            lines.append(f"   • {msg}")
        if len(high) > 5:
            lines.append(f"   *...and {len(high) - 5} more*")
        lines.append("")

    if medium:
        lines.append(f"🟡 **MEDIUM: {len(medium)}**")
        for a in medium[:3]:
            msg = str(a.get("message", ""))[:80]
            lines.append(f"   • {msg}")
        if len(medium) > 3:
            lines.append(f"   *...and {len(medium) - 3} more*")

    if info:
        lines.append(f"ℹ️ **INFO: {len(info)}**")

    return "\n".join(lines)


def _handle_top_products(df: pd.DataFrame) -> str:
    """Handle top products queries."""
    sort_col = None
    for col in ["expected_revenue", "revenue"]:
        if col in df.columns:
            sort_col = col
            break

    if sort_col is None:
        if "current_price" in df.columns and "sales_volume" in df.columns:
            df = df.copy()
            df["_est_revenue"] = df["current_price"] * df["sales_volume"]
            sort_col = "_est_revenue"
        else:
            return "⚠️ Revenue data not available for sorting."

    total = min(10, len(df))
    top = df.nlargest(total, sort_col)
    lines = []
    lines.append(_section_header("🏆 **TOP PRODUCTS BY REVENUE**"))
    lines.append("")
    lines.append("")

    for i, (_, r) in enumerate(top.iterrows(), 1):
        name = str(r.get("product_name", "Unknown"))[:50]
        revenue = r.get(sort_col, 0)
        lines.append(f"**{i}. {name}**")
        lines.append(f"   • **Revenue:** {_fmtd(revenue)}")

        for col in ["margin_percentage", "profit_margin"]:
            if col in r.index:
                margin = r[col] * 100 if r[col] < 1 and col == "profit_margin" else r[col]
                lines.append(f"   • **Margin:** {margin:.1f}%")
                break

        if "composite_risk_score" in r.index:
            risk = r["composite_risk_score"]
            icon = "🟢" if risk < 30 else "🟡" if risk < 60 else "🟠" if risk < 80 else "🔴"
            lines.append(f"   • **Risk:** {icon} {risk:.0f}/100")

        lines.append("")

    return "\n".join(lines)


def _handle_help() -> str:
    """Handle help queries."""
    return (
        "## 🤖 **AI Business Assistant — Help**\n\n"
        "You can ask me questions about your pricing data in natural language.\n\n"
        "### **PRICING & REPRICING**\n"
        "• \"Which products need repricing?\"\n"
        "• \"Where should we increase prices?\"\n"
        "• \"Which products need markdowns?\"\n\n"
        "### **PROFIT & MARGINS**\n"
        "• \"Show margin opportunities.\"\n"
        "• \"Which products have low margins?\"\n"
        "• \"Biggest profit potential?\"\n\n"
        "### **RISK ASSESSMENT**\n"
        "• \"Which products are high risk?\"\n"
        "• \"Show risky products.\"\n"
        "• \"Risk summary.\"\n\n"
        "### **INVENTORY & STOCK**\n"
        "• \"Inventory status.\"\n"
        "• \"Which products are overstocked?\"\n"
        "• \"Low stock items?\"\n\n"
        "### **DEMAND & FORECASTING**\n"
        "• \"Demand forecast.\"\n"
        "• \"Sales predictions.\"\n"
        "• \"Future demand trends?\"\n\n"
        "### **COMPETITORS & MARKET**\n"
        "• \"How are we positioned?\"\n"
        "• \"Competitor analysis.\"\n"
        "• \"Pricing gaps?\"\n\n"
        "### **PORTFOLIO & ALERTS**\n"
        "• \"Portfolio overview.\"\n"
        "• \"Show active alerts.\"\n"
        "• \"Top products by revenue.\""
    )


def _handle_generic_fallback() -> str:
    """Generic fallback response."""
    return (
        "I didn't fully understand that question. Here's what I can help with:\n\n"
        "• **Pricing and repricing** opportunities\n"
        "• **Margin and profit** analysis\n"
        "• **Risk assessment**\n"
        "• **Inventory** status\n"
        "• **Demand forecasting**\n"
        "• **Competitive** positioning\n"
        "• **Portfolio** overview\n\n"
        "Type **\"help\"** for example questions."
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ASSISTANT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AIAssistant:
    """
    AI Business Analyst assistant powered by OpenAI (with rule-based fallback).

    Connects to the pricing platform's analysis data and answers natural
    language business questions with concise, data-driven responses.

    Supports:
    - OpenAI integration (gpt-4o-mini) when API key is configured
    - Rule-based fallback for all common queries (no API key needed)
    - Conversation memory (last 10 turns)
    - Live DataFrame context for accurate answers
    - Business-focused query routing

    Usage:
        assistant = AIAssistant(api_key="sk-...")  # OpenAI key (optional)
        assistant.load_context(df, insights, alerts)

        # Ask questions
        response = assistant.ask("Which products need repricing?")
        print(response)

        # With conversation history
        assistant.ask("Show risky products")
        assistant.ask("What about the Electronics category?")  # Has memory of previous turn
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        api_key: Optional[str] = None,
        use_openai: bool = True,
        model: str = "gpt-4o-mini",
    ) -> None:
        """
        Initialize the AI assistant.

        Args:
            config: Application configuration.
            api_key: OpenAI API key (or set OPENAI_API_KEY env var).
            use_openai: Whether to attempt OpenAI calls (falls back to rules if fails).
            model: OpenAI model name.
        """
        self.config = config or AppConfig()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = use_openai and OPENAI_AVAILABLE and bool(self.api_key)
        self.model = model

        self._df: Optional[pd.DataFrame] = None
        self._insights: Dict[str, Any] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._context_built: bool = False
        self._system_context: str = ""

        # Conversation memory
        self._history: List[ChatMessage] = []
        self._last_context_refresh: Optional[datetime] = None

        if self.use_openai:
            logger.info(f"AI Assistant initialised with OpenAI ({model})")
        else:
            reason = "package not installed" if not OPENAI_AVAILABLE else "no API key"
            logger.info(f"AI Assistant initialised in rule-only mode ({reason})")

    @property
    def history(self) -> List[ChatMessage]:
        """Get the conversation history."""
        return self._history

    def load_context(
        self,
        df: pd.DataFrame,
        insights: Dict[str, Any],
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Load analysis context for the assistant to query.

        Builds the system prompt context string from the provided data.

        Args:
            df: Full analysis DataFrame.
            insights: Summary insights dictionary.
            alerts: Optional list of generated alerts.
        """
        safe_df = _normalise_assistant_df(df)
        self._df = safe_df
        self._insights = insights or {}
        self._alerts = alerts or []
        self._system_context = _build_system_context(safe_df, self._insights, self._alerts) if not safe_df.empty else ""
        self._context_built = True
        self._last_context_refresh = datetime.now()

        logger.info(
            f"Context loaded: {len(safe_df)} products, "
            f"{len(self._alerts)} alerts"
        )

    def refresh_context(self) -> None:
        """Rebuild the system context from the current data."""
        if self._df is not None:
            self._system_context = _build_system_context(
                self._df, self._insights, self._alerts
            )
            self._context_built = True
            self._last_context_refresh = datetime.now()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def ask(
        self,
        query: str,
        use_openai: Optional[bool] = None,
    ) -> str:
        """
        Ask a business question and get a data-driven response.

        Auto-detects the intent and routes to the appropriate handler.
        Tries OpenAI first (if enabled and available), falls back to
        rule-based response.

        Args:
            query: Natural language question.
            use_openai: Override whether to use OpenAI for this query.

        Returns:
            Response text string.
        """
        if not self._context_built or self._df is None:
            return (
                "⚠️ No data loaded. Please load a product dataset and run "
                "the analysis pipeline first."
            )

        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # Add to history
        self._history.append(ChatMessage(
            role="user",
            content=query_stripped,
            timestamp=datetime.now().isoformat(),
        ))

        # Trim history
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

        response: Optional[str] = None

        # Try OpenAI first
        should_use_openai = use_openai if use_openai is not None else self.use_openai
        if should_use_openai:
            response = self._ask_openai(query_stripped)

        # Fallback to rules
        if response is None:
            response = _fallback_response(query_stripped, self._df, self._insights, self._alerts)
        response = _clean_response_start(response)

        # Add to history
        self._history.append(ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
        ))

        return response

    def _ask_openai(self, query: str) -> Optional[str]:
        """
        Send the query to OpenAI with system context and conversation history.

        Args:
            query: User's question.

        Returns:
            Response text or None if call fails.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_context},
        ]

        # Add recent history (last 3 turns)
        for msg in self._history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current query
        messages.append({"role": "user", "content": query})

        return _call_openai(
            messages=messages,
            api_key=self.api_key,
            model=self.model,
        )

    def ask_streamlit(
        self,
        query: str,
        use_openai: Optional[bool] = None,
    ) -> str:
        """
        Streamlit-ready alias for ask(). Returns formatted markdown string.

        Args:
            query: User's question.
            use_openai: Whether to use OpenAI.

        Returns:
            Markdown-formatted response.
        """
        return self.ask(query, use_openai=use_openai)

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get the conversation history as a DataFrame.

        Returns:
            DataFrame with role, content, and timestamp columns.
        """
        return pd.DataFrame([m.to_dict() for m in self._history])
