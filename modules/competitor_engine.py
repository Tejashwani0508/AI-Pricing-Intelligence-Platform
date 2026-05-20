"""
Competitor Intelligence Engine Module

Analyses competitive positioning for every product by comparing our price
against competitor prices and market averages. Detects pricing anomalies,
competitive threats, and generates actionable intelligence.

Features:
  - Overpriced detection:    our price significantly above market
  - Underpriced detection:   our price significantly below market (margin leakage)
  - Competitor undercutting: competitor prices dropping relative to ours
  - Aggressive competition:  market-wide price pressure in a category

Output per product:
  - competitive_score       (0-100, higher = stronger position)
  - price_ratio             (our_price / competitor_price)
  - market_position         (Premium / Competitive / Discount / Aggressive Discount)
  - overpriced_flag         (bool)
  - underpriced_flag        (bool)
  - competitor_undercut_flag (bool)
  - aggressive_market_flag  (bool)
  - pricing_recommendation  (actionable text)
  - price_gap_pct           (% difference from competitor)
  - price_vs_market         (our_price / market_average)

All logic is vectorised for 1000+ products.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import AppConfig
from utils.helpers import safe_divide, format_currency

logger = logging.getLogger("ai_pricing.competitor_engine")


# ─── Constants ─────────────────────────────────────────────────────────────

# Price ratio thresholds for market position
PREMIUM_THRESHOLD: float = 1.15       # 15% above competitor → Premium
COMPETITIVE_UPPER: float = 1.05       # 5% above → upper Competitive
COMPETITIVE_LOWER: float = 0.95       # 5% below → lower Competitive
DISCOUNT_THRESHOLD: float = 0.85      # 15% below → Discount (else Aggressive)

# Overpriced / underpriced detection
OVERRICED_RATIO: float = 1.20         # 20% above competitor → overpriced
UNDERPRICED_RATIO: float = 0.80       # 20% below competitor → underpriced

# Undercut detection
UNDERCUT_THRESHOLD: float = 0.90      # Competitor ≤ 90% of our price → undercut

# Aggressive market competition
AGGRESSIVE_CATEGORY_DISCOUNT_SHARE: float = 0.30  # ≥30% of category at Discount or below

# Default competitive score weights
SCORE_WEIGHTS: Dict[str, float] = {
    "margin_health": 0.25,
    "price_positioning": 0.30,
    "demand_strength": 0.20,
    "market_share": 0.15,
    "price_stability": 0.10,
}


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class CompetitiveAnalysis:
    """
    Full competitive analysis result for a single product.
    """
    product_id: str = ""
    product_name: str = ""
    our_price: float = 0.0
    competitor_price: float = 0.0
    market_average: float = 0.0
    price_ratio: float = 1.0
    price_gap_pct: float = 0.0
    price_vs_market: float = 1.0
    market_position: str = "Competitive"
    competitive_score: float = 50.0
    overpriced_flag: bool = False
    underpriced_flag: bool = False
    competitor_undercut_flag: bool = False
    aggressive_market_flag: bool = False
    pricing_recommendation: str = ""
    market_share_estimate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "our_price": round(self.our_price, 2),
            "competitor_price": round(self.competitor_price, 2),
            "market_average": round(self.market_average, 2),
            "price_ratio": round(self.price_ratio, 3),
            "price_gap_pct": round(self.price_gap_pct, 2),
            "price_vs_market": round(self.price_vs_market, 3),
            "market_position": self.market_position,
            "competitive_score": round(self.competitive_score, 1),
            "overpriced": self.overpriced_flag,
            "underpriced": self.underpriced_flag,
            "competitor_undercut": self.competitor_undercut_flag,
            "aggressive_market": self.aggressive_market_flag,
            "recommendation": self.pricing_recommendation,
        }


@dataclass
class CompetitorReport:
    """
    Aggregate report for a batch competitive analysis.
    """
    total_products: int = 0
    overpriced_count: int = 0
    underpriced_count: int = 0
    undercut_count: int = 0
    aggressive_market_count: int = 0
    avg_competitive_score: float = 0.0
    avg_price_ratio: float = 1.0
    position_distribution: Dict[str, int] = field(default_factory=dict)
    top_overpriced: List[Dict[str, Any]] = field(default_factory=list)
    top_undercut: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_products,
            "overpriced": self.overpriced_count,
            "underpriced": self.underpriced_count,
            "competitor_undercutting": self.undercut_count,
            "aggressive_markets": self.aggressive_market_count,
            "avg_competitive_score": round(self.avg_competitive_score, 1),
            "avg_price_ratio": round(self.avg_price_ratio, 3),
            "position_distribution": self.position_distribution,
            "recommendations": self.recommendation_summary,
        }

    def summary(self) -> str:
        return (
            f"  Products:             {self.total_products}\n"
            f"  Overpriced:           {self.overpriced_count}\n"
            f"  Underpriced:          {self.underpriced_count}\n"
            f"  Competitor Undercut:  {self.undercut_count}\n"
            f"  Aggressive Markets:   {self.aggressive_market_count}\n"
            f"  Avg Competitive Score: {self.avg_competitive_score:.1f}/100\n"
            f"  Avg Price Ratio:      {self.avg_price_ratio:.3f}x"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS (vectorised)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_price_metrics(
    our_price: np.ndarray,
    competitor_price: np.ndarray,
    market_average: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fundamental price comparison metrics.

    Args:
        our_price: Array of our current prices.
        competitor_price: Array of competitor prices.
        market_average: Array of market average prices (uses competitor if None).

    Returns:
        Tuple of (price_ratio, price_gap_pct, price_vs_market, market_avg_used).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        price_ratio = np.where(
            competitor_price > 0,
            our_price / competitor_price,
            1.0,
        )
        price_gap_pct = (price_ratio - 1.0) * 100

    if market_average is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            price_vs_market = np.where(
                market_average > 0,
                our_price / market_average,
                1.0,
            )
        market_avg = market_average
    else:
        price_vs_market = price_ratio.copy()
        market_avg = competitor_price.copy()

    return price_ratio, price_gap_pct, price_vs_market, market_avg


def _classify_market_position(price_ratio: np.ndarray) -> np.ndarray:
    """
    Classify each product's market position based on price ratio.

    Returns:
        Array of position strings.
    """
    conditions = [
        price_ratio >= PREMIUM_THRESHOLD,
        price_ratio >= COMPETITIVE_UPPER,
        price_ratio >= COMPETITIVE_LOWER,
        price_ratio >= DISCOUNT_THRESHOLD,
    ]
    choices = ["Premium", "Competitive", "Competitive", "Discount"]
    return np.select(conditions, choices, default="Aggressive Discount")


def _detect_overpriced(
    price_ratio: np.ndarray,
    profit_margin: np.ndarray,
    market_position: np.ndarray,
) -> np.ndarray:
    """
    Detect products that are overpriced relative to the market.

    Conditions:
    - Price ratio > 20% above competitor (OVERRICED_RATIO)
    - OR Premium position with low margin (< 15%)

    Returns:
        Boolean array.
    """
    over_by_price = price_ratio > OVERRICED_RATIO
    over_by_position = (
        (market_position == "Premium") & (profit_margin < 0.15)
    )
    return over_by_price | over_by_position


def _detect_underpriced(
    price_ratio: np.ndarray,
    profit_margin: np.ndarray,
    market_position: np.ndarray,
) -> np.ndarray:
    """
    Detect products that are underpriced (margin leakage).

    Conditions:
    - Price ratio < 20% below competitor (UNDERPRICED_RATIO)
    - OR Aggressive Discount with healthy margin (> 25%)

    Returns:
        Boolean array.
    """
    under_by_price = price_ratio < UNDERPRICED_RATIO
    under_by_position = (
        (market_position == "Aggressive Discount") & (profit_margin > 0.25)
    )
    return under_by_price | under_by_position


def _detect_competitor_undercutting(
    price_ratio: np.ndarray,
    competitor_price: np.ndarray,
    our_price: np.ndarray,
) -> np.ndarray:
    """
    Detect where competitors are actively undercutting our price.

    A product is "undercut" when the competitor's price is at least
    10% below our price (price_ratio > 1.10 means we're >10% above them).

    Returns:
        Boolean array.
    """
    return price_ratio > (1.0 / UNDERCUT_THRESHOLD)


def _detect_aggressive_market(
    category: np.ndarray,
    market_position: np.ndarray,
    df_index: pd.Index,
) -> np.ndarray:
    """
    Detect categories where a significant share of products are
    in Discount or Aggressive Discount positions — indicating
    aggressive market-wide competition.

    Args:
        category: Array of category strings.
        market_position: Array of position strings.
        df_index: Original DataFrame index for reconstruction.

    Returns:
        Boolean array indicating aggressive-market products.
    """
    # Build a temporary Series for groupby
    cat_series = pd.Series(category, index=df_index)
    pos_series = pd.Series(market_position, index=df_index)

    # Per category: share of products in Discount or Aggressive Discount
    is_discount = pos_series.isin(["Discount", "Aggressive Discount"])
    discount_share = is_discount.groupby(cat_series).transform("mean")

    return discount_share.values >= AGGRESSIVE_CATEGORY_DISCOUNT_SHARE


def _compute_competitive_score(
    profit_margin: np.ndarray,
    price_ratio: np.ndarray,
    market_position: np.ndarray,
    demand_trend: np.ndarray,
    market_share: np.ndarray,
    price_vs_market: np.ndarray,
) -> np.ndarray:
    """
    Compute a competitive score (0-100) for each product.

    Higher score = stronger competitive position.

    Components:
    1. Margin Health (0-25): higher margin → higher score
    2. Price Positioning (0-30): competitive position → best score
    3. Demand Strength (0-20): higher demand → higher score
    4. Market Share (0-15): higher share → higher score
    5. Price Stability (0-10): close to 1.0 ratio → stable

    Args:
        profit_margin: Profit margin (decimal).
        price_ratio: Our price / competitor price.
        market_position: Position classification string.
        demand_trend: Demand signal (0-1).
        market_share: Estimated market share (0-100).
        price_vs_market: Our price / market average.

    Returns:
        Array of competitive scores (0-100).
    """
    n = len(profit_margin)
    scores = np.zeros(n, dtype=float)

    # 1. Margin Health (0-25)
    margin_score = np.clip(profit_margin * 25, 0, 25)

    # 2. Price Positioning (0-30)
    # Competitive → 30, Premium → 20, Discount → 15, Aggressive → 5
    pos_score = np.select(
        [market_position == "Competitive",
         market_position == "Premium",
         market_position == "Discount"],
        [30.0, 20.0, 15.0],
        default=5.0,
    )

    # 3. Demand Strength (0-20)
    demand_score = demand_trend * 20

    # 4. Market Share (0-15) — use percentile within dataset
    if n > 1:
        share_pct_rank = pd.Series(market_share).rank(pct=True).values
        share_score = share_pct_rank * 15
    else:
        share_score = np.array([7.5])

    # 5. Price Stability (0-10) — closer to 1.0 = more stable
    with np.errstate(divide="ignore", invalid="ignore"):
        # Volatility: how far from 1.0
        vol = np.abs(price_vs_market - 1.0)
        stability_score = np.clip((1.0 - vol / 0.3) * 10, 0, 10)

    scores = margin_score + pos_score + demand_score + share_score + stability_score
    return np.clip(scores, 0, 100)


def _generate_recommendation(
    overpriced: np.ndarray,
    underpriced: np.ndarray,
    undercut: np.ndarray,
    aggressive: np.ndarray,
    market_position: np.ndarray,
    price_gap_pct: np.ndarray,
    product_names: np.ndarray,
) -> List[str]:
    """
    Generate actionable pricing recommendations for each product.

    Args:
        overpriced: Boolean flag.
        underpriced: Boolean flag.
        undercut: Boolean flag.
        aggressive: Boolean flag.
        market_position: Position string.
        price_gap_pct: Price gap percentage.
        product_names: Product name strings.

    Returns:
        List of recommendation strings.
    """
    n = len(overpriced)
    recommendations: List[str] = []

    for i in range(n):
        parts: List[str] = []

        if overpriced[i]:
            parts.append(
                f"Reduce price by {abs(price_gap_pct[i]):.0f}% "
                f"to align with market (currently overpriced)"
            )

        if underpriced[i]:
            parts.append(
                f"Increase price by {abs(price_gap_pct[i]):.0f}% — "
                f"current price leaves margin on the table"
            )

        if undercut[i] and not overpriced[i]:
            parts.append(
                f"Competitor is undercutting by "
                f"{abs(price_gap_pct[i]):.0f}%; consider value differentiation"
            )

        if aggressive[i] and market_position[i] == "Competitive":
            parts.append(
                f"Category is highly competitive; monitor margins closely"
            )

        if not parts:
            if market_position[i] == "Competitive":
                parts.append("Maintain current pricing strategy")
            elif market_position[i] == "Premium":
                parts.append("Maintain premium position; ensure value delivery")
            elif market_position[i] == "Discount":
                parts.append("Continue discount strategy; watch margins")
            else:
                parts.append("Review pricing strategy for this product")

        recommendations.append(" | ".join(parts))

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION (standalone)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_competition(
    df: pd.DataFrame,
    our_price_col: str = "current_price",
    competitor_price_col: str = "competitor_price",
    market_avg_col: Optional[str] = None,
    margin_col: Optional[str] = None,
    demand_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    category_col: str = "category",
    name_col: str = "product_name",
    inplace: bool = False,
) -> Tuple[pd.DataFrame, CompetitorReport]:
    """
    Run full competitive intelligence analysis on all products.

    Computes price ratios, market positions, detects overpriced/underpriced
    products, competitor undercutting, aggressive market competition, and
    generates competitive scores with actionable recommendations.

    Args:
        df: DataFrame with product data.
        our_price_col: Column with our prices.
        competitor_price_col: Column with competitor prices.
        market_avg_col: Optional column with market average prices.
        margin_col: Optional column with profit margin (0-1).
        demand_col: Optional column with demand trend (0-1).
        volume_col: Optional column with sales volume.
        category_col: Column with product category.
        name_col: Column with product name.
        inplace: If True, modifies df in place.

    Returns:
        Tuple of:
        - Enriched DataFrame with 15+ competitive analysis columns
        - CompetitorReport with aggregate statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if not inplace:
        result = df.copy()
    else:
        result = df

    report = CompetitorReport()
    report.total_products = len(df)

    # ─── Validate required columns ────────────────────────────────────
    required = [our_price_col, competitor_price_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    n = len(df)
    logger.info(f"analyze_competition: {n} products")

    # ─── Extract arrays ───────────────────────────────────────────────
    our_price = pd.to_numeric(df[our_price_col], errors="coerce").fillna(0).values
    competitor_fallback = pd.Series(our_price * 1.05, index=df.index)
    competitor_price = pd.to_numeric(
        df[competitor_price_col], errors="coerce"
    ).fillna(competitor_fallback).values

    # Market average: use provided column or competitor price
    if market_avg_col and market_avg_col in df.columns:
        market_average = pd.to_numeric(
            df[market_avg_col], errors="coerce"
        ).fillna(competitor_price).values
    else:
        market_average = competitor_price.copy()

    # Margin
    if margin_col and margin_col in df.columns:
        profit_margin = pd.to_numeric(
            df[margin_col], errors="coerce"
        ).fillna(0).values
    else:
        # Compute from available columns
        if "profit_margin" in df.columns:
            profit_margin = pd.to_numeric(
                df["profit_margin"], errors="coerce"
            ).fillna(0).values
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                profit_margin = np.where(
                    our_price > 0,
                    (our_price - pd.to_numeric(
                        df["cost_price"], errors="coerce"
                    ).fillna(0).values) / our_price,
                    0.0,
                )

    # Demand
    if demand_col and demand_col in df.columns:
        demand_trend = pd.to_numeric(
            df[demand_col], errors="coerce"
        ).fillna(0.5).values
    elif "demand_trend" in df.columns:
        demand_trend = pd.to_numeric(
            df["demand_trend"], errors="coerce"
        ).fillna(0.5).values
    else:
        demand_trend = np.full(n, 0.5)

    # Volume for market share
    if volume_col and volume_col in df.columns:
        sales_volume = pd.to_numeric(
            df[volume_col], errors="coerce"
        ).fillna(0).values
    elif "sales_volume" in df.columns:
        sales_volume = pd.to_numeric(
            df["sales_volume"], errors="coerce"
        ).fillna(0).values
    else:
        sales_volume = np.full(n, 100)

    # Category
    category = df[category_col].values if category_col in df.columns else np.full(n, "default")

    # Product names
    product_names = df[name_col].values if name_col in df.columns else np.full(n, "")

    # ─── Compute metrics ──────────────────────────────────────────────
    price_ratio, price_gap_pct, price_vs_market, market_avg = _compute_price_metrics(
        our_price, competitor_price, market_average
    )

    market_position = _classify_market_position(price_ratio)

    # Overpriced / underpriced
    overpriced = _detect_overpriced(price_ratio, profit_margin, market_position)
    underpriced = _detect_underpriced(price_ratio, profit_margin, market_position)

    # Competitor undercutting
    undercut = _detect_competitor_undercutting(price_ratio, competitor_price, our_price)

    # Aggressive market competition
    aggressive_market = _detect_aggressive_market(
        category, market_position, df.index
    )

    # Market share (within category)
    # Compute market share properly
    if category_col in df.columns and volume_col in df.columns:
        cat_totals_series = df.groupby(category_col)[volume_col].transform("sum")
        market_share = (sales_volume / cat_totals_series.values) * 100
        market_share = np.nan_to_num(market_share, nan=0.0, posinf=1.0, neginf=0.0)
    else:
        market_share = np.full(n, 1.0)

    # Competitive score
    competitive_score = _compute_competitive_score(
        profit_margin, price_ratio, market_position,
        demand_trend, market_share, price_vs_market,
    )

    # Pricing recommendations
    recommendations = _generate_recommendation(
        overpriced, underpriced, undercut, aggressive_market,
        market_position, price_gap_pct, product_names,
    )

    # ─── Write results to DataFrame ───────────────────────────────────
    result["price_ratio"] = np.round(price_ratio, 3)
    result["price_gap_pct"] = np.round(price_gap_pct, 2)
    result["price_vs_market"] = np.round(price_vs_market, 3)
    result["market_average"] = np.round(market_average, 2)
    result["market_position"] = market_position
    result["competitive_score"] = np.round(competitive_score, 1)
    result["overpriced_flag"] = overpriced
    result["underpriced_flag"] = underpriced
    result["competitor_undercut_flag"] = undercut
    result["aggressive_market_flag"] = aggressive_market
    result["pricing_recommendation"] = recommendations
    result["estimated_market_share"] = np.round(market_share, 2)

    # ─── Build report ─────────────────────────────────────────────────
    report.overpriced_count = int(np.sum(overpriced))
    report.underpriced_count = int(np.sum(underpriced))
    report.undercut_count = int(np.sum(undercut))
    report.aggressive_market_count = int(np.sum(aggressive_market))
    report.avg_competitive_score = float(np.nanmean(competitive_score))
    report.avg_price_ratio = float(np.nanmean(price_ratio))

    # Position distribution
    pos_series = pd.Series(market_position)
    report.position_distribution = pos_series.value_counts().to_dict()

    # Top overpriced
    overpriced_indices = np.where(overpriced)[0]
    if len(overpriced_indices) > 0:
        sorted_over = overpriced_indices[np.argsort(price_gap_pct[overpriced_indices])[::-1]][:5]
        for idx in sorted_over:
            report.top_overpriced.append({
                "product_name": str(product_names[idx]) if idx < len(product_names) else "",
                "price_gap_pct": round(float(price_gap_pct[idx]), 1),
                "competitive_score": round(float(competitive_score[idx]), 1),
            })

    # Top undercut
    undercut_indices = np.where(undercut)[0]
    if len(undercut_indices) > 0:
        sorted_under = undercut_indices[np.argsort(price_gap_pct[undercut_indices])[::-1]][:5]
        for idx in sorted_under:
            report.top_undercut.append({
                "product_name": str(product_names[idx]) if idx < len(product_names) else "",
                "price_gap_pct": round(float(price_gap_pct[idx]), 1),
                "competitive_score": round(float(competitive_score[idx]), 1),
            })

    # Recommendation summary
    rec_series = pd.Series(recommendations)
    for prefix in ["Maintain", "Reduce", "Increase", "Review", "Continue", "Competitor", "Category"]:
        count = int(rec_series.str.startswith(prefix).sum())
        if count > 0:
            report.recommendation_summary[prefix] = count

    logger.info(
        f"Analysis complete: {report.overpriced_count} overpriced, "
        f"{report.underpriced_count} underpriced, "
        f"{report.undercut_count} undercut, "
        f"{report.aggressive_market_count} aggressive markets, "
        f"avg_score={report.avg_competitive_score:.1f}"
    )

    return result, report


# ═══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def batch_competitor_analysis(
    df: pd.DataFrame,
    chunk_size: int = 5000,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, CompetitorReport]:
    """
    Run competitor analysis in chunks for very large datasets.

    Args:
        df: Product DataFrame.
        chunk_size: Products per chunk (default 5000).
        **kwargs: Additional arguments passed to analyze_competition().

    Returns:
        Tuple of (enriched DataFrame, merged CompetitorReport).
    """
    total = len(df)
    logger.info(f"Batch competitor analysis: {total} products, chunk={chunk_size}")

    if total <= chunk_size:
        return analyze_competition(df, **kwargs)

    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, total, chunk_size)]
    chunk_results: List[pd.DataFrame] = []
    combined_report = CompetitorReport()
    combined_report.total_products = total

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_result, chunk_report = analyze_competition(chunk, inplace=True, **kwargs)
        chunk_results.append(chunk_result)

        combined_report.overpriced_count += chunk_report.overpriced_count
        combined_report.underpriced_count += chunk_report.underpriced_count
        combined_report.undercut_count += chunk_report.undercut_count
        combined_report.aggressive_market_count += chunk_report.aggressive_market_count

    result_df = pd.concat(chunk_results, ignore_index=True)
    combined_report.avg_competitive_score = float(result_df["competitive_score"].mean())
    combined_report.avg_price_ratio = float(result_df["price_ratio"].mean())
    combined_report.position_distribution = result_df["market_position"].value_counts().to_dict()

    return result_df, combined_report


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API
# ═══════════════════════════════════════════════════════════════════════════

class CompetitorEngine:
    """
    Competitor intelligence engine.

    Analyses every product's competitive position by comparing our price
    against competitor prices and market averages. Detects overpriced and
    underpriced products, competitor undercutting, and aggressive market
    competition. Generates competitive scores (0-100) and actionable
    pricing recommendations.

    All logic is vectorised for 1000+ products.

    Usage:
        engine = CompetitorEngine()
        df, report = engine.analyze(dataframe)
        print(report.summary())
        print(df["pricing_recommendation"].iloc[0])
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the competitor engine.

        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()
        self._last_report: Optional[CompetitorReport] = None

    @property
    def last_report(self) -> Optional[CompetitorReport]:
        """Get the report from the most recent analysis."""
        return self._last_report

    def analyze(
        self,
        df: pd.DataFrame,
        our_price_col: str = "current_price",
        competitor_price_col: str = "competitor_price",
        market_avg_col: Optional[str] = None,
        category_col: str = "category",
        inplace: bool = False,
    ) -> Tuple[pd.DataFrame, CompetitorReport]:
        """
        Run competitive intelligence analysis on all products.

        Args:
            df: Product DataFrame.
            our_price_col: Our price column.
            competitor_price_col: Competitor price column.
            market_avg_col: Optional market average column.
            category_col: Category column.
            inplace: If True, modifies df in place.

        Returns:
            Tuple of (enriched DataFrame, CompetitorReport).
        """
        result_df, report = analyze_competition(
            df,
            our_price_col=our_price_col,
            competitor_price_col=competitor_price_col,
            market_avg_col=market_avg_col,
            category_col=category_col,
            inplace=inplace,
        )
        self._last_report = report
        return result_df, report

    def analyze_single(
        self,
        our_price: float,
        competitor_price: float,
        market_average: Optional[float] = None,
        profit_margin: Optional[float] = None,
        demand_trend: Optional[float] = None,
        sales_volume: Optional[float] = None,
        category: str = "default",
        product_name: str = "",
    ) -> CompetitiveAnalysis:
        """
        Analyse competitive position for a single product.

        Args:
            our_price: Our current price.
            competitor_price: Average competitor price.
            market_average: Market average price.
            profit_margin: Profit margin (0-1).
            demand_trend: Demand signal (0-1).
            sales_volume: Monthly sales volume.
            category: Product category.
            product_name: Product name.

        Returns:
            CompetitiveAnalysis with results.
        """
        row = {
            "current_price": our_price,
            "competitor_price": competitor_price,
            "product_name": product_name or "Product",
        }
        if market_average is not None:
            row["market_average"] = market_average
        if profit_margin is not None:
            row["profit_margin"] = profit_margin
        if demand_trend is not None:
            row["demand_trend"] = demand_trend
        if sales_volume is not None:
            row["sales_volume"] = sales_volume
        row["category"] = category

        df = pd.DataFrame([row])
        result_df, _ = self.analyze(df, inplace=False)

        if result_df.empty:
            return CompetitiveAnalysis(product_name=product_name)

        row_out = result_df.iloc[0]

        return CompetitiveAnalysis(
            product_id=str(row_out.get("product_id", "")),
            product_name=str(row_out.get("product_name", "")),
            our_price=float(row_out.get("current_price", 0)),
            competitor_price=float(row_out.get("competitor_price", 0)),
            market_average=float(row_out.get("market_average", 0)),
            price_ratio=float(row_out.get("price_ratio", 1.0)),
            price_gap_pct=float(row_out.get("price_gap_pct", 0)),
            price_vs_market=float(row_out.get("price_vs_market", 1.0)),
            market_position=str(row_out.get("market_position", "Competitive")),
            competitive_score=float(row_out.get("competitive_score", 50)),
            overpriced_flag=bool(row_out.get("overpriced_flag", False)),
            underpriced_flag=bool(row_out.get("underpriced_flag", False)),
            competitor_undercut_flag=bool(row_out.get("competitor_undercut_flag", False)),
            aggressive_market_flag=bool(row_out.get("aggressive_market_flag", False)),
            pricing_recommendation=str(row_out.get("pricing_recommendation", "")),
            market_share_estimate=float(row_out.get("estimated_market_share", 0)),
        )

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last analysis."""
        if self._last_report is None:
            return "No analysis performed yet."
        return self._last_report.summary()
