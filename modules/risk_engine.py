"""
Risk Scoring Engine Module

Generates explainable 0-100 risk scores for each product across five factors:

  1. Demand Risk     — low demand, declining trend, high elasticity
  2. Profitability Risk — falling or negative margins, cost pressure
  3. Inventory Risk  — excess stock, low turnover, obsolescence
  4. Competitor Risk — priced above competitor, losing position
  5. Margin Risk     — weak absolute margins, erosion

Composite score = weighted sum of all factor scores.
Each factor includes a human-readable reason string.

Output per product:
  risk_score        (0-100)
  risk_level        (LOW / MEDIUM / HIGH / CRITICAL)
  risk_reason       (concatenated explanation of contributing factors)
  (plus per-factor scores for explainability)

All operations are vectorised for 1000+ products.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.config import AppConfig
from utils.helpers import safe_divide

logger = logging.getLogger("ai_pricing.risk_engine")


# ─── Constants ─────────────────────────────────────────────────────────────

# Risk level thresholds
LOW_BOUNDARY: float = 30.0
MEDIUM_BOUNDARY: float = 50.0
HIGH_BOUNDARY: float = 70.0

# Default factor weights (sum = 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "demand": 0.20,
    "profitability": 0.25,
    "inventory": 0.15,
    "competitor": 0.20,
    "margin": 0.20,
}

# Risk level labels ordered by severity
RISK_LEVELS: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class RiskFactorScore:
    """
    Score and explanation for a single risk factor.
    """
    name: str = ""
    score: float = 0.0          # 0–100 sub-score for this factor
    weight: float = 0.0         # Contribution weight in composite
    contribution: float = 0.0   # Weighted contribution to composite (score × weight)
    reason: str = ""            # Human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "contribution": round(self.contribution, 1),
            "reason": self.reason,
        }


@dataclass
class ProductRisk:
    """
    Complete risk assessment for a single product.
    """
    product_id: str = ""
    product_name: str = ""
    composite_score: float = 0.0     # 0–100
    risk_level: str = "LOW"
    risk_reason: str = ""
    factors: Dict[str, RiskFactorScore] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "risk_score": round(self.composite_score, 1),
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "factor_details": {
                k: v.to_dict() for k, v in self.factors.items()
            },
        }


@dataclass
class RiskReport:
    """
    Aggregate risk report for a batch of products.
    """
    total_products: int = 0
    products_with_scores: int = 0
    products_with_errors: int = 0

    level_counts: Dict[str, int] = field(default_factory=lambda: {
        "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0,
    })
    avg_composite_score: float = 0.0
    max_composite_score: float = 0.0
    min_composite_score: float = 0.0

    avg_factor_scores: Dict[str, float] = field(default_factory=dict)
    primary_risk_factor_distribution: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    top_high_risk: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_products": self.total_products,
            "scored": self.products_with_scores,
            "errors": self.products_with_errors,
            "level_counts": self.level_counts,
            "avg_risk_score": round(self.avg_composite_score, 1),
            "risk_range": [
                round(self.min_composite_score, 1),
                round(self.max_composite_score, 1),
            ],
            "avg_factor_scores": {
                k: round(v, 1) for k, v in self.avg_factor_scores.items()
            },
            "primary_risk_distribution": self.primary_risk_factor_distribution,
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        high_crit = self.level_counts.get("HIGH", 0) + self.level_counts.get("CRITICAL", 0)
        factor_lines = "\n".join(
            f"    {k}: {v:.1f}"
            for k, v in sorted(self.avg_factor_scores.items(), key=lambda x: -x[1])
        )
        return (
            f"  Products scored:  {self.products_with_scores} / {self.total_products}\n"
            f"  Errors:           {self.products_with_errors}\n"
            f"  Avg risk score:   {self.avg_composite_score:.1f}\n"
            f"  HIGH / CRITICAL:  {high_crit}\n"
            f"  Risk Levels:\n"
            f"    CRITICAL: {self.level_counts.get('CRITICAL', 0)}\n"
            f"    HIGH:     {self.level_counts.get('HIGH', 0)}\n"
            f"    MEDIUM:   {self.level_counts.get('MEDIUM', 0)}\n"
            f"    LOW:      {self.level_counts.get('LOW', 0)}\n"
            f"  Avg Factor Scores:\n"
            f"{factor_lines}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTOR SCORING FUNCTIONS  (each returns 0-100 + reason)
# ═══════════════════════════════════════════════════════════════════════════

def _score_demand_risk(
    demand_trend: np.ndarray,
    sales_volume: np.ndarray,
    price_elasticity: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Score demand-side risk (0-100).

    Factors:
    - Low demand_trend (< 0.5) → higher risk
    - Low sales volume (relative to dataset) → higher risk
    - High price elasticity (|e| > 1.5) → higher risk

    Returns:
        Tuple of (score array 0-100, list of reason strings per product).
    """
    n = len(demand_trend)
    score = np.zeros(n, dtype=float)
    reasons: List[str] = []

    # --- Demand Trend Component (0-40 points) ---
    # demand_trend < 0.3 → full 40; 0.3–0.7 → scaled; > 0.7 → 0
    trend_risk = np.where(
        demand_trend < 0.3,
        40.0,
        np.where(
            demand_trend < 0.7,
            (0.7 - demand_trend) / 0.4 * 40.0,
            0.0,
        ),
    )

    # --- Sales Volume Component (0-35 points) ---
    # Rank-based: bottom 20% percentile → high risk
    # Use percentile rank within the dataset
    if n > 1:
        vol_rank = pd.Series(sales_volume).rank(pct=True).values
        vol_risk = np.where(
            vol_rank < 0.2,
            35.0,
            np.where(
                vol_rank < 0.5,
                (0.5 - vol_rank) / 0.3 * 35.0,
                0.0,
            ),
        )
    else:
        vol_risk = np.zeros(n)

    # --- Elasticity Component (0-25 points) ---
    abs_elast = np.abs(price_elasticity)
    elast_risk = np.where(
        abs_elast >= 2.5,
        25.0,
        np.where(
            abs_elast >= 1.5,
            (abs_elast - 1.5) / 1.0 * 25.0,
            0.0,
        ),
    )

    score = trend_risk + vol_risk + elast_risk
    score = np.clip(score, 0, 100)

    # Build reasons per product
    for i in range(n):
        parts = []
        if trend_risk[i] > 15:
            parts.append(f"low demand trend ({demand_trend[i]:.2f})")
        if vol_risk[i] > 15:
            parts.append(f"low sales volume ({sales_volume[i]:.0f})")
        if elast_risk[i] > 10:
            parts.append(f"high elasticity ({abs_elast[i]:.1f})")
        reasons.append("; ".join(parts) if parts else "")

    return score, reasons


def _score_profitability_risk(
    current_price: np.ndarray,
    cost_price: np.ndarray,
    profit_margin: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Score profitability risk (0-100).

    Factors:
    - Negative or very low margin → high risk
    - High cost-to-price ratio → cost pressure
    - Margin below 10% → flagged

    Returns:
        Tuple of (score array 0-100, reason strings).
    """
    n = len(current_price)
    score = np.zeros(n, dtype=float)
    reasons: List[str] = []

    # --- Margin Level Component (0-60 points) ---
    # margin < 0% → 60; 0-10% → scaled; 10-25% → low; > 25% → 0
    with np.errstate(divide="ignore", invalid="ignore"):
        margin_risk = np.where(
            profit_margin <= 0.0,
            60.0,
            np.where(
                profit_margin < 0.10,
                (0.10 - profit_margin) / 0.10 * 60.0,
                np.where(
                    profit_margin < 0.25,
                    (0.25 - profit_margin) / 0.15 * 30.0,
                    0.0,
                ),
            ),
        )

    # --- Cost Ratio Component (0-40 points) ---
    # cost_price / current_price > 0.85 → high risk
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_ratio = np.where(
            current_price > 0,
            cost_price / current_price,
            0.0,
        )
    cost_risk = np.where(
        cost_ratio >= 0.95,
        40.0,
        np.where(
            cost_ratio >= 0.85,
            (cost_ratio - 0.85) / 0.10 * 40.0,
            np.where(
                cost_ratio >= 0.70,
                (cost_ratio - 0.70) / 0.15 * 20.0,
                0.0,
            ),
        ),
    )

    score = margin_risk + cost_risk
    score = np.clip(score, 0, 100)

    for i in range(n):
        parts = []
        marg_pct = profit_margin[i] * 100
        if margin_risk[i] > 20:
            parts.append(f"low margin ({marg_pct:.1f}%)")
        if cost_risk[i] > 15:
            parts.append(f"high cost ratio ({cost_ratio[i]*100:.0f}%)")
        reasons.append("; ".join(parts) if parts else "")

    return score, reasons


def _score_inventory_risk(
    inventory_level: np.ndarray,
    sales_volume: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Score inventory risk (0-100).

    Factors:
    - Excess stock (high days of cover or low turnover)
    - Very low stock (stockout risk)
    - Slow-moving inventory

    Returns:
        Tuple of (score array 0-100, reason strings).
    """
    n = len(inventory_level)
    score = np.zeros(n, dtype=float)
    reasons: List[str] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        # Days of cover proxy: inventory / (sales/30)
        daily_sales = sales_volume / 30.0
        days_cover = np.where(
            daily_sales > 0,
            inventory_level / daily_sales,
            np.where(inventory_level > 0, 999.0, 0.0),
        )

    # --- Excess Stock Component (0-55 points) ---
    # days_cover > 180 → high excess risk
    excess_risk = np.where(
        days_cover >= 365,
        55.0,
        np.where(
            days_cover >= 180,
            (days_cover - 180) / 185 * 55.0,
            np.where(
                days_cover >= 90,
                (days_cover - 90) / 90 * 30.0,
                0.0,
            ),
        ),
    )

    # --- Stockout Risk Component (0-30 points) ---
    # Very low inventory relative to sales
    stockout_risk = np.where(
        (inventory_level < 10) & (sales_volume > 0),
        30.0,
        np.where(
            days_cover < 7,
            (7 - days_cover) / 7 * 30.0,
            0.0,
        ),
    )

    # --- Turnover Component (0-15 points) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = np.where(
            inventory_level > 0,
            sales_volume / inventory_level,
            0.0,
        )
    turnover_risk = np.where(
        (turnover < 0.5) & (inventory_level > 0),
        15.0,
        np.where(
            turnover < 1.0,
            (1.0 - turnover) / 0.5 * 10.0,
            0.0,
        ),
    )

    score = excess_risk + stockout_risk + turnover_risk
    score = np.clip(score, 0, 100)

    for i in range(n):
        parts = []
        if excess_risk[i] > 15:
            parts.append(f"excess stock (~{days_cover[i]:.0f}d cover)")
        if stockout_risk[i] > 15:
            parts.append(f"low stock ({inventory_level[i]:.0f} units)")
        if turnover_risk[i] > 5:
            parts.append(f"slow turnover ({turnover[i]:.1f}x)")
        reasons.append("; ".join(parts) if parts else "")

    return score, reasons


def _score_competitor_risk(
    current_price: np.ndarray,
    competitor_price: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Score competitive pressure risk (0-100).

    Factors:
    - Priced above competitor → losing price advantage
    - Large price gap (above competitor) → risk of losing customers

    Returns:
        Tuple of (score array 0-100, reason strings).
    """
    n = len(current_price)
    score = np.zeros(n, dtype=float)
    reasons: List[str] = []

    with np.errstate(divide="ignore", invalid="ignore"):
        price_ratio = np.where(
            competitor_price > 0,
            current_price / competitor_price,
            1.0,
        )
        price_gap_pct = (price_ratio - 1.0) * 100  # positive = we're more expensive

    # --- Price Above Competitor Component (0-70 points) ---
    # > 25% more expensive → full 70; 10-25% → scaled; < 10% → 0
    above_risk = np.where(
        price_ratio >= 1.25,
        70.0,
        np.where(
            price_ratio >= 1.10,
            (price_ratio - 1.10) / 0.15 * 70.0,
            0.0,
        ),
    )

    # --- Price Below Competitor (discount) Component (0-30 points) ---
    # Being significantly cheaper may also indicate quality concerns or race to bottom
    below_risk = np.where(
        price_ratio <= 0.75,
        30.0,
        np.where(
            price_ratio <= 0.85,
            (0.85 - price_ratio) / 0.10 * 30.0,
            0.0,
        ),
    )

    score = above_risk + below_risk
    score = np.clip(score, 0, 100)

    for i in range(n):
        parts = []
        if above_risk[i] > 15:
            parts.append(f"priced {price_gap_pct[i]:.0f}% above competitor")
        if below_risk[i] > 15:
            parts.append(f"priced {abs(price_gap_pct[i]):.0f}% below competitor (race to bottom)")
        reasons.append("; ".join(parts) if parts else "")

    return score, reasons


def _score_margin_weakness_risk(
    profit_margin: np.ndarray,
    current_price: np.ndarray,
    cost_price: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Score margin weakness risk (0-100).

    This is a more targeted assessment of margin health:
    - Margin below 10% → high risk
    - Negative margin → critical risk
    - Thin absolute dollar margin → risk

    Returns:
        Tuple of (score array 0-100, reason strings).
    """
    n = len(current_price)
    score = np.zeros(n, dtype=float)
    reasons: List[str] = []

    # --- Margin Percentage Component (0-60 points) ---
    margin_pct = profit_margin * 100  # convert to percentage
    pct_risk = np.where(
        margin_pct <= 0,
        60.0,
        np.where(
            margin_pct < 5,
            (5 - margin_pct) / 5 * 60.0,
            np.where(
                margin_pct < 10,
                (10 - margin_pct) / 5 * 40.0,
                np.where(
                    margin_pct < 20,
                    (20 - margin_pct) / 10 * 20.0,
                    0.0,
                ),
            ),
        ),
    )

    # --- Absolute Dollar Margin Component (0-40 points) ---
    # Even if margin % is ok, low absolute profit per unit is a risk
    abs_margin = current_price - cost_price
    dollar_risk = np.where(
        abs_margin <= 0,
        40.0,
        np.where(
            abs_margin < 5,
            (5 - abs_margin) / 5 * 40.0,
            np.where(
                abs_margin < 10,
                (10 - abs_margin) / 5 * 20.0,
                0.0,
            ),
        ),
    )

    score = pct_risk + dollar_risk
    score = np.clip(score, 0, 100)

    for i in range(n):
        parts = []
        if pct_risk[i] > 20:
            parts.append(f"weak margin ({margin_pct[i]:.1f}%)")
        if dollar_risk[i] > 15:
            parts.append(f"low per-unit profit (${abs_margin[i]:.2f})")
        reasons.append("; ".join(parts) if parts else "")

    return score, reasons


def _determine_risk_level(score: float) -> str:
    """Map a 0-100 score to a risk level string."""
    if score >= HIGH_BOUNDARY:
        return "CRITICAL" if score >= 85 else "HIGH"
    elif score >= MEDIUM_BOUNDARY:
        return "MEDIUM"
    else:
        return "LOW"


def _build_composite_reason(factors: Dict[str, RiskFactorScore]) -> str:
    """
    Build a composite risk reason string by concatenating the top contributing factors.

    Args:
        factors: Dict of factor name -> RiskFactorScore.

    Returns:
        Human-readable reason string.
    """
    # Sort factors by contribution (highest first)
    sorted_factors = sorted(
        factors.values(),
        key=lambda f: f.contribution,
        reverse=True,
    )

    # Take factors that contribute meaningfully
    active = [f for f in sorted_factors if f.score >= 20 and f.reason]
    if not active:
        return "No significant risk factors detected."

    # Build concise summary
    parts = []
    for f in active[:3]:  # max 3 reasons
        weight_pct = f.weight * 100
        parts.append(f"{f.reason} (factor contribution: {f.contribution:.0f}/100, weight: {weight_pct:.0f}%)")

    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RISK ASSESSMENT FUNCTION (Standalone)
# ═══════════════════════════════════════════════════════════════════════════

def assess_risks(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, RiskReport]:
    """
    Perform full risk assessment on all products using vectorised operations.

    Computes five factor scores (0-100 each), a weighted composite score (0-100),
    a risk level (LOW / MEDIUM / HIGH / CRITICAL), and a human-readable
    risk reason explaining the top contributing factors.

    Args:
        df: DataFrame with at minimum current_price and cost_price.
            Optional but recommended: sales_volume, inventory_level,
            demand_trend, price_elasticity, competitor_price.
        weights: Dict of factor weights (defaults to DEFAULT_WEIGHTS).
        inplace: If True, modifies df in place; otherwise returns a copy.

    Returns:
        Tuple of:
        - Enriched DataFrame with columns:
            demand_risk_score, profitability_risk_score, inventory_risk_score,
            competitor_risk_score, margin_risk_score,
            composite_risk_score, risk_level, risk_reason
        - RiskReport with aggregate statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not inplace:
        result = df.copy()
    else:
        result = df

    report = RiskReport()
    report.total_products = len(df)
    n = len(df)

    # ─── Validate and prepare inputs ───────────────────────────────────
    required = ["current_price", "cost_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Ensure numeric
    current_price = pd.to_numeric(df["current_price"], errors="coerce").fillna(0).values
    cost_price = pd.to_numeric(df["cost_price"], errors="coerce").fillna(0).values

    # Compute profit_margin if not present
    if "profit_margin" in df.columns:
        profit_margin = pd.to_numeric(df["profit_margin"], errors="coerce").fillna(0).values
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            profit_margin = np.where(
                current_price > 0,
                (current_price - cost_price) / current_price,
                0.0,
            )

    # Optional columns with defaults
    if "demand_trend" in df.columns:
        demand_trend = pd.to_numeric(df["demand_trend"], errors="coerce").fillna(0.5).values
    else:
        demand_trend = np.full(n, 0.5)

    if "sales_volume" in df.columns:
        sales_volume = pd.to_numeric(df["sales_volume"], errors="coerce").fillna(0).clip(0).values
    else:
        sales_volume = np.full(n, 100)

    if "price_elasticity" in df.columns:
        price_elasticity = pd.to_numeric(df["price_elasticity"], errors="coerce").fillna(-1.0).values
    else:
        price_elasticity = np.full(n, -1.0)

    if "inventory_level" in df.columns:
        inventory_level = pd.to_numeric(df["inventory_level"], errors="coerce").fillna(0).clip(0).values
    else:
        inventory_level = np.full(n, 500)

    if "competitor_price" in df.columns:
        competitor_price = (
            pd.to_numeric(df["competitor_price"], errors="coerce")
            .fillna(pd.Series(current_price * 1.05, index=df.index))
            .values
        )
    else:
        competitor_price = current_price * 1.05

    logger.info(
        f"assess_risks: {n} products, "
        f"weights={weights}"
    )

    factor_names = ["demand", "profitability", "inventory", "competitor", "margin"]
    factor_weight_list = [weights.get(f, 0.0) for f in factor_names]

    try:
        # ─── Compute factor scores ─────────────────────────────────────
        demand_scores, demand_reasons = _score_demand_risk(
            demand_trend, sales_volume, price_elasticity
        )
        profitability_scores, profitability_reasons = _score_profitability_risk(
            current_price, cost_price, profit_margin
        )
        inventory_scores, inventory_reasons = _score_inventory_risk(
            inventory_level, sales_volume
        )
        competitor_scores, competitor_reasons = _score_competitor_risk(
            current_price, competitor_price
        )
        margin_scores, margin_reasons = _score_margin_weakness_risk(
            profit_margin, current_price, cost_price
        )

        # Stack into matrix: shape (n_products, 5)
        all_scores = np.column_stack([
            demand_scores,
            profitability_scores,
            inventory_scores,
            competitor_scores,
            margin_scores,
        ])
        all_weights = np.array(factor_weight_list, dtype=float)

        # Weighted composite score (vectorised)
        composite_scores = all_scores @ all_weights  # dot product
        composite_scores = np.clip(composite_scores, 0, 100)

        # ─── Build per-product risk assessment ─────────────────────────
        risk_levels: List[str] = []
        risk_reasons: List[str] = []
        top_factors_list: List[str] = []
        product_risk_objs: List[ProductRisk] = []

        for i in range(n):
            pid = str(df.iloc[i].get("product_id", "")) if "product_id" in df.columns else ""
            pname = str(df.iloc[i].get("product_name", "")) if "product_name" in df.columns else ""

            factors_dict: Dict[str, RiskFactorScore] = {}
            factor_scores_list = [
                ("demand", demand_scores[i], demand_reasons[i]),
                ("profitability", profitability_scores[i], profitability_reasons[i]),
                ("inventory", inventory_scores[i], inventory_reasons[i]),
                ("competitor", competitor_scores[i], competitor_reasons[i]),
                ("margin", margin_scores[i], margin_reasons[i]),
            ]

            for fname, fscore, freason in factor_scores_list:
                fweight = weights.get(fname, 0.0)
                factors_dict[fname] = RiskFactorScore(
                    name=fname,
                    score=float(fscore),
                    weight=fweight,
                    contribution=float(fscore * fweight),
                    reason=freason if freason else "",
                )

            composite = float(composite_scores[i])
            level = _determine_risk_level(composite)
            reason = _build_composite_reason(factors_dict)

            # Identify primary risk factor
            sorted_by_contrib = sorted(
                factors_dict.values(),
                key=lambda f: f.contribution,
                reverse=True,
            )
            top_factor = sorted_by_contrib[0].name if sorted_by_contrib else "unknown"
            top_factors_list.append(top_factor)

            risk_levels.append(level)
            risk_reasons.append(reason)

            product_risk = ProductRisk(
                product_id=pid,
                product_name=pname,
                composite_score=composite,
                risk_level=level,
                risk_reason=reason,
                factors=factors_dict,
            )
            product_risk_objs.append(product_risk)

        # ─── Write results to DataFrame ────────────────────────────────
        result["demand_risk_score"] = np.round(demand_scores, 1)
        result["profitability_risk_score"] = np.round(profitability_scores, 1)
        result["inventory_risk_score"] = np.round(inventory_scores, 1)
        result["competitor_risk_score"] = np.round(competitor_scores, 1)
        result["margin_risk_score"] = np.round(margin_scores, 1)
        result["composite_risk_score"] = np.round(composite_scores, 1)
        result["risk_level"] = risk_levels
        result["risk_reason"] = risk_reasons
        result["primary_risk_factor"] = top_factors_list

        # ─── Build aggregate report ────────────────────────────────────
        report.products_with_scores = int(np.isfinite(composite_scores).sum())
        report.products_with_errors = n - report.products_with_scores

        for level in RISK_LEVELS:
            report.level_counts[level] = int(sum(1 for l in risk_levels if l == level))

        report.avg_composite_score = float(np.nanmean(composite_scores))
        report.max_composite_score = float(np.nanmax(composite_scores))
        report.min_composite_score = float(np.nanmin(composite_scores))

        report.avg_factor_scores = {
            "demand": float(np.nanmean(demand_scores)),
            "profitability": float(np.nanmean(profitability_scores)),
            "inventory": float(np.nanmean(inventory_scores)),
            "competitor": float(np.nanmean(competitor_scores)),
            "margin": float(np.nanmean(margin_scores)),
        }

        # Primary risk factor distribution
        pf_counts = pd.Series(top_factors_list).value_counts()
        report.primary_risk_factor_distribution = pf_counts.to_dict()

        # Top 5 high-risk products
        high_risk_mask = composite_scores >= HIGH_BOUNDARY
        if high_risk_mask.any():
            high_indices = np.argsort(-composite_scores)[:5]
            for idx in high_indices:
                if composite_scores[idx] >= HIGH_BOUNDARY:
                    report.top_high_risk.append({
                        "product_id": str(df.iloc[idx].get("product_id", "")),
                        "product_name": str(df.iloc[idx].get("product_name", "")),
                        "risk_score": round(float(composite_scores[idx]), 1),
                        "risk_level": risk_levels[idx],
                        "primary_factor": top_factors_list[idx],
                    })

        logger.info(
            f"Risk assessment complete: {report.products_with_scores} scored, "
            f"avg={report.avg_composite_score:.1f}, "
            f"HIGH/CRITICAL={report.level_counts.get('HIGH',0)+report.level_counts.get('CRITICAL',0)}, "
            f"primary factors={report.primary_risk_factor_distribution}"
        )

    except Exception as e:
        logger.error(f"Risk assessment error: {e}", exc_info=True)
        report.errors.append(str(e))

        # Write NaN columns on failure so pipeline doesn't break
        for col in [
            "demand_risk_score", "profitability_risk_score",
            "inventory_risk_score", "competitor_risk_score",
            "margin_risk_score", "composite_risk_score",
            "risk_level", "risk_reason", "primary_risk_factor",
        ]:
            result[col] = np.nan if col != "risk_level" else "ERROR"

    return result, report


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API  (backward-compatible wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Multi-factor risk assessment engine.

    Computes explainable 0-100 risk scores for each product across five
    dimensions: demand, profitability, inventory, competitor, and margin.

    Each factor produces a sub-score with a human-readable reason.
    The composite score is a weighted sum, mapped to a risk level.

    Risk Levels:  LOW (0-29) | MEDIUM (30-49) | HIGH (50-69) | CRITICAL (70-100)

    Usage:
        engine = RiskEngine()
        df, report = engine.assess_risk(dataframe)
        print(report.summary())
        print(df["risk_reason"].iloc[0])  # explainable reason
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the risk engine.

        Args:
            config: Application configuration (for thresholds).
            weights: Custom factor weights (uses defaults if None).
        """
        self.config = config or AppConfig()
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._last_report: Optional[RiskReport] = None

    @property
    def last_report(self) -> Optional[RiskReport]:
        """Get the report from the most recent risk assessment."""
        return self._last_report

    def assess_risk(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
        **weight_overrides: float,
    ) -> Tuple[pd.DataFrame, RiskReport]:
        """
        Assess risk for all products in the DataFrame.

        Args:
            df: Product DataFrame with at least 'current_price' and 'cost_price'.
            inplace: If True, modifies df in place.
            **weight_overrides: Override factor weights
                                (e.g., demand=0.30, margin=0.15).

        Returns:
            Tuple of (enriched DataFrame, RiskReport).
        """
        # Apply weight overrides
        if weight_overrides:
            weights = self.weights.copy()
            weights.update(weight_overrides)
        else:
            weights = self.weights

        result_df, report = assess_risks(df, weights=weights, inplace=inplace)
        self._last_report = report
        return result_df, report

    def assess_single(
        self,
        current_price: float,
        cost_price: float,
        demand_trend: Optional[float] = None,
        sales_volume: Optional[float] = None,
        inventory_level: Optional[float] = None,
        competitor_price: Optional[float] = None,
        price_elasticity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Assess risk for a single product.

        Args:
            current_price: Current selling price.
            cost_price: Unit cost.
            demand_trend: Demand signal (0-1).
            sales_volume: Units sold.
            inventory_level: Current stock.
            competitor_price: Average competitor price.
            price_elasticity: Price sensitivity.

        Returns:
            Dict with risk assessment results and factor breakdown.
        """
        row: Dict[str, Any] = {
            "current_price": current_price,
            "cost_price": cost_price,
        }
        if demand_trend is not None:
            row["demand_trend"] = demand_trend
        if sales_volume is not None:
            row["sales_volume"] = sales_volume
        if inventory_level is not None:
            row["inventory_level"] = inventory_level
        if competitor_price is not None:
            row["competitor_price"] = competitor_price
        if price_elasticity is not None:
            row["price_elasticity"] = price_elasticity

        df = pd.DataFrame([row])
        result_df, _ = self.assess_risk(df, inplace=False)

        if result_df.empty:
            return {"error": "No result produced"}

        row_out = result_df.iloc[0].to_dict()

        # Build explainable dict
        response = {
            "composite_risk_score": row_out.get("composite_risk_score"),
            "risk_level": row_out.get("risk_level"),
            "risk_reason": row_out.get("risk_reason"),
            "factor_scores": {
                "demand": row_out.get("demand_risk_score"),
                "profitability": row_out.get("profitability_risk_score"),
                "inventory": row_out.get("inventory_risk_score"),
                "competitor": row_out.get("competitor_risk_score"),
                "margin": row_out.get("margin_risk_score"),
            },
            "primary_risk_factor": row_out.get("primary_risk_factor"),
        }
        # Remove None values
        return {k: v for k, v in response.items() if v is not None}

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last risk assessment."""
        if self._last_report is None:
            return "No risk assessment performed yet."
        return self._last_report.summary()
