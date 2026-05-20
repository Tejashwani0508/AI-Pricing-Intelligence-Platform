"""
Explainability Module

Generates human-readable explanations for pricing recommendations
for every product. Each explanation breaks down the key factors
driving the optimal price recommendation into clear bullet points.

Example Output:
  Recommended Price: $64.00 (change: -20.0%)
  Reasons:
    • strong demand  (demand_trend=0.85 → +15% premium)
    • low inventory  (15 days cover → +8% scarcity premium)
    • competitor priced higher (competitor=$74.99 → anchor to +11%)
    • margin improvement needed (current 12.3% → target 20.0%)

Supports:
  - generate_explanation()        — single product
  - batch_explanation()           — all products in a DataFrame
  - DataFrame integration          — adds explanation column(s)
  - Template-based reasoning       — deterministic, auditable
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import AppConfig
from utils.helpers import safe_divide, format_currency

logger = logging.getLogger("ai_pricing.explainability")


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class ExplanationFactor:
    """
    A single factor that contributed to a pricing recommendation.
    """
    label: str = ""              # e.g. "strong demand"
    detail: str = ""             # e.g. "demand_trend=0.85 → +15% premium"
    direction: str = "neutral"   # "upward", "downward", "neutral"
    contribution_pct: float = 0.0  # estimated % impact on price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "detail": self.detail,
            "direction": self.direction,
            "contribution_pct": round(self.contribution_pct, 1),
        }


@dataclass
class PriceExplanation:
    """
    Complete pricing explanation for a single product.
    """
    product_id: str = ""
    product_name: str = ""
    current_price: float = 0.0
    optimal_price: float = 0.0
    price_change_pct: float = 0.0
    recommendation: str = ""
    summary: str = ""                # One-line summary
    bullet_points: List[str] = field(default_factory=list)  # Formatted bullets
    factors: List[ExplanationFactor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "current_price": round(self.current_price, 2),
            "optimal_price": round(self.optimal_price, 2),
            "price_change_pct": round(self.price_change_pct, 1),
            "recommendation": self.recommendation,
            "summary": self.summary,
            "bullet_points": self.bullet_points,
            "factors": [f.to_dict() for f in self.factors],
        }

    def to_text(self) -> str:
        """Render as a human-readable text block."""
        lines = [
            f"Product: {self.product_name} ({self.product_id})",
            f"Current Price: ${self.current_price:.2f}",
            f"Recommended Price: ${self.optimal_price:.2f} ({self.price_change_pct:+.1f}%)",
            f"Recommendation: {self.recommendation}",
            f"Reasons:",
        ]
        for bullet in self.bullet_points:
            lines.append(f"  • {bullet}")
        return "\n".join(lines)


@dataclass
class ExplanationReport:
    """
    Aggregate report for a batch explanation run.
    """
    total_products: int = 0
    explained: int = 0
    errors: int = 0
    error_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_products,
            "explained": self.explained,
            "errors": self.errors,
            "error_ids": self.error_ids[:10],
        }

    def summary(self) -> str:
        return (
            f"  Products:  {self.total_products}\n"
            f"  Explained: {self.explained}\n"
            f"  Errors:    {self.errors}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTOR ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _analyze_demand_factor(
    demand_trend: float,
    price_elasticity: float,
) -> ExplanationFactor:
    """
    Analyse how demand influences the pricing recommendation.

    Returns:
        ExplanationFactor with direction and estimated contribution.
    """
    factor = ExplanationFactor(label="demand")
    abs_elast = abs(price_elasticity)

    if demand_trend >= 0.7:
        premium = min(0.15, (demand_trend - 0.7) / 0.3 * 0.15)
        factor.direction = "upward"
        factor.contribution_pct = premium * 100
        factor.label = "strong demand"
        factor.detail = (
            f"demand_trend={demand_trend:.2f} "
            f"(above 0.7 threshold) → +{premium*100:.0f}% premium"
        )
    elif demand_trend >= 0.5:
        factor.direction = "neutral"
        factor.contribution_pct = 0.0
        factor.label = "moderate demand"
        factor.detail = f"demand_trend={demand_trend:.2f} (neutral range)"
    else:
        discount = min(0.10, (0.5 - demand_trend) / 0.5 * 0.10)
        factor.direction = "downward"
        factor.contribution_pct = -discount * 100
        factor.label = "weak demand"
        factor.detail = (
            f"demand_trend={demand_trend:.2f} "
            f"(below 0.5) → -{discount*100:.0f}% discount"
        )

    # Price elasticity modifier
    if abs_elast >= 2.0 and factor.direction == "upward":
        factor.detail += "; high elasticity limits upside"
        factor.contribution_pct *= 0.7  # dampen
    elif abs_elast <= 1.0 and factor.direction == "downward":
        factor.detail += "; inelastic demand limits discount impact"
        factor.contribution_pct *= 0.7  # dampen

    return factor


def _analyze_inventory_factor(
    inventory_level: float,
    sales_volume: float,
) -> ExplanationFactor:
    """
    Analyse how inventory levels influence pricing.

    Returns:
        ExplanationFactor.
    """
    factor = ExplanationFactor(label="inventory")

    with np.errstate(divide="ignore", invalid="ignore"):
        days_cover = (
            (inventory_level / sales_volume * 30)
            if sales_volume > 0
            else 999.0
        )

    if days_cover < 15:
        scarcity = min(0.08, (15 - days_cover) / 15 * 0.08)
        factor.direction = "upward"
        factor.contribution_pct = scarcity * 100
        factor.label = "low inventory"
        factor.detail = (
            f"~{days_cover:.0f} days cover "
            f"(below 15 threshold) → +{scarcity*100:.0f}% scarcity premium"
        )
    elif days_cover > 90:
        excess = min(0.05, (days_cover - 90) / 90 * 0.05)
        factor.direction = "downward"
        factor.contribution_pct = -excess * 100
        factor.label = "excess inventory"
        factor.detail = (
            f"~{days_cover:.0f} days cover "
            f"(above 90 threshold) → -{excess*100:.0f}% excess discount"
        )
    else:
        factor.direction = "neutral"
        factor.contribution_pct = 0.0
        factor.label = "healthy inventory"
        factor.detail = f"~{days_cover:.0f} days cover (healthy range)"

    return factor


def _analyze_competitor_factor(
    current_price: float,
    competitor_price: float,
) -> ExplanationFactor:
    """
    Analyse how competitor pricing influences the recommendation.

    Returns:
        ExplanationFactor.
    """
    factor = ExplanationFactor(label="competitor")

    with np.errstate(divide="ignore", invalid="ignore"):
        price_ratio = (
            current_price / competitor_price
            if competitor_price > 0
            else 1.0
        )
        gap_pct = (price_ratio - 1.0) * 100

    if price_ratio > 1.10:
        # We're priced above competitor → downward pressure
        pressure = min(0.15, (price_ratio - 1.10) / 0.15 * 0.15)
        factor.direction = "downward"
        factor.contribution_pct = -pressure * 100
        factor.label = "competitor priced lower"
        factor.detail = (
            f"our price {gap_pct:+.0f}% above competitor "
            f"(${competitor_price:.2f}) → -{pressure*100:.0f}% competitive adjustment"
        )
    elif price_ratio < 0.90:
        # We're priced below competitor → room to increase
        room = min(0.10, (0.90 - price_ratio) / 0.10 * 0.10)
        factor.direction = "upward"
        factor.contribution_pct = room * 100
        factor.label = "competitor priced higher"
        factor.detail = (
            f"our price {abs(gap_pct):.0f}% below competitor "
            f"(${competitor_price:.2f}) → +{room*100:.0f}% room to increase"
        )
    else:
        factor.direction = "neutral"
        factor.contribution_pct = 0.0
        factor.label = "competitively priced"
        factor.detail = (
            f"price ratio {price_ratio:.2f}x "
            f"(within ±10% of competitor ${competitor_price:.2f})"
        )

    return factor


def _analyze_margin_factor(
    current_price: float,
    cost_price: float,
    optimal_price: float,
) -> ExplanationFactor:
    """
    Analyse how margin considerations influence the recommendation.

    Returns:
        ExplanationFactor.
    """
    factor = ExplanationFactor(label="margin")

    with np.errstate(divide="ignore", invalid="ignore"):
        current_margin = (
            (current_price - cost_price) / current_price
            if current_price > 0
            else 0.0
        )
        optimal_margin = (
            (optimal_price - cost_price) / optimal_price
            if optimal_price > 0
            else 0.0
        )

    margin_pct = current_margin * 100
    optimal_margin_pct = optimal_margin * 100

    if current_margin < 0.10:
        # Very low margin → strong upward pressure
        needed = min(0.20, (0.10 - current_margin) / 0.10 * 0.20)
        factor.direction = "upward"
        factor.contribution_pct = needed * 100
        factor.label = "margin improvement needed"
        factor.detail = (
            f"current margin {margin_pct:.1f}% is below 10% threshold; "
            f"optimal raises to {optimal_margin_pct:.1f}%"
        )
    elif current_margin < 0.20:
        # Moderate margin → slight upward pressure
        needed = min(0.05, (0.20 - current_margin) / 0.10 * 0.05)
        factor.direction = "upward"
        factor.contribution_pct = needed * 100
        factor.label = "margin optimisation"
        factor.detail = (
            f"current margin {margin_pct:.1f}% below 20% target; "
            f"optimal raises to {optimal_margin_pct:.1f}%"
        )
    elif current_margin >= 0.40:
        # High margin → room to discount if needed
        factor.direction = "downward"
        factor.contribution_pct = -3.0  # small fixed discount
        factor.label = "healthy margin"
        factor.detail = (
            f"current margin {margin_pct:.1f}% is strong "
            f"(above 40%); provides pricing flexibility"
        )
    else:
        factor.direction = "neutral"
        factor.contribution_pct = 0.0
        factor.label = "adequate margin"
        factor.detail = f"current margin {margin_pct:.1f}% is within acceptable range"

    return factor


def _analyze_cost_factor(
    current_price: float,
    cost_price: float,
    optimal_price: float,
) -> ExplanationFactor:
    """
    Analyse how cost pressure influences the recommendation.

    Returns:
        ExplanationFactor.
    """
    factor = ExplanationFactor(label="cost")

    with np.errstate(divide="ignore", invalid="ignore"):
        cost_ratio = cost_price / current_price if current_price > 0 else 0.0
        cost_pct = cost_ratio * 100

    if cost_ratio >= 0.90:
        factor.direction = "upward"
        factor.contribution_pct = 10.0
        factor.label = "high cost pressure"
        factor.detail = (
            f"cost is {cost_pct:.0f}% of price (${cost_price:.2f} cost "
            f"vs ${current_price:.2f} price); "
            f"price increase needed to maintain margin"
        )
    elif cost_ratio >= 0.75:
        factor.direction = "slight_upward"
        factor.contribution_pct = 5.0
        factor.label = "moderate cost ratio"
        factor.detail = (
            f"cost is {cost_pct:.0f}% of price; "
            f"monitor for margin erosion"
        )
    else:
        factor.direction = "neutral"
        factor.contribution_pct = 0.0
        factor.label = "low cost ratio"
        factor.detail = (
            f"cost is {cost_pct:.0f}% of price; "
            f"healthy cost structure"
        )

    return factor


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPLANATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_explanation(
    product_data: Dict[str, Any],
) -> PriceExplanation:
    """
    Generate a human-readable pricing explanation for a single product.

    Analyses demand, inventory, competitor, margin, and cost factors,
    then produces a structured explanation with bullet points.

    Args:
        product_data: Dict with product fields including:
            - product_id, product_name
            - current_price, optimal_price, cost_price
            - competitor_price, demand_trend, sales_volume,
              inventory_level, price_elasticity
            - recommendation (optional)

    Returns:
        PriceExplanation with summary, bullet_points, and factors.

    Raises:
        ValueError: If required fields are missing.
    """
    required = ["current_price", "cost_price"]
    missing = [r for r in required if r not in product_data]
    if missing:
        raise ValueError(
            f"Missing required fields: {missing}. "
            f"Got: {list(product_data.keys())}"
        )

    # Extract fields with defaults
    pid = str(product_data.get("product_id", ""))
    pname = str(product_data.get("product_name", ""))
    current_price = float(product_data.get("current_price", 0))
    cost_price = float(product_data.get("cost_price", 0))

    # Use optimal_price if provided, otherwise compute a target
    if "optimal_price" in product_data and pd.notna(product_data.get("optimal_price")):
        optimal_price = float(product_data["optimal_price"])
    else:
        # Fallback: use current price (no optimisation data)
        optimal_price = current_price

    competitor_price = float(product_data.get("competitor_price", current_price * 1.05))
    demand_trend = float(product_data.get("demand_trend", 0.5))
    sales_volume = float(product_data.get("sales_volume", 100))
    inventory_level = float(product_data.get("inventory_level", 500))
    price_elasticity = float(product_data.get("price_elasticity", -1.5))

    if "recommendation" in product_data:
        recommendation = str(product_data["recommendation"])
    else:
        # Determine from price change
        with np.errstate(divide="ignore", invalid="ignore"):
            change_pct = (
                (optimal_price - current_price) / current_price * 100
                if current_price > 0
                else 0.0
            )
        if change_pct > 2.0:
            recommendation = "Increase"
        elif change_pct < -2.0:
            recommendation = "Decrease"
        else:
            recommendation = "Maintain"

    with np.errstate(divide="ignore", invalid="ignore"):
        price_change_pct = (
            (optimal_price - current_price) / current_price * 100
            if current_price > 0
            else 0.0
        )

    # ═══ Analyse all factors ═══════════════════════════════════════════
    factors: List[ExplanationFactor] = []
    factor_list = [
        _analyze_demand_factor(demand_trend, price_elasticity),
        _analyze_inventory_factor(inventory_level, sales_volume),
        _analyze_competitor_factor(current_price, competitor_price),
        _analyze_margin_factor(current_price, cost_price, optimal_price),
        _analyze_cost_factor(current_price, cost_price, optimal_price),
    ]

    # Filter out neutral factors with zero contribution
    for f in factor_list:
        if f.direction != "neutral" or f.contribution_pct != 0.0:
            # Only include non-trivial factors
            if abs(f.contribution_pct) >= 1.0 or f.direction != "neutral":
                factors.append(f)

    # Sort by absolute contribution (most impactful first)
    factors.sort(key=lambda f: abs(f.contribution_pct), reverse=True)

    # Limit to top 5 factors
    top_factors = factors[:5]

    # ═══ Build bullet points ═══════════════════════════════════════════
    # Direction icons
    icons = {
        "upward": "↑",
        "slight_upward": "↗",
        "downward": "↓",
        "slight_downward": "↘",
        "neutral": "→",
    }

    bullets: List[str] = []
    for f in top_factors:
        icon = icons.get(f.direction, "→")
        contrib_str = (
            f" ({f.contribution_pct:+.0f}% effect)"
            if abs(f.contribution_pct) >= 1.0
            else ""
        )
        bullets.append(f"{icon} {f.label}: {f.detail}{contrib_str}")

    if not bullets:
        bullets.append("→ All factors neutral; maintain current pricing.")

    # One-line summary
    n_factors = len(top_factors)
    summary_parts = []
    for f in top_factors[:3]:
        summary_parts.append(f.label)
    summary = (
        f"{recommendation} price to ${optimal_price:.2f} "
        f"({price_change_pct:+.1f}%) driven by "
        f"{', '.join(summary_parts)}"
    )

    return PriceExplanation(
        product_id=pid,
        product_name=pname,
        current_price=current_price,
        optimal_price=optimal_price,
        price_change_pct=price_change_pct,
        recommendation=recommendation,
        summary=summary,
        bullet_points=bullets,
        factors=top_factors,
    )


def batch_explanation(
    df: pd.DataFrame,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, ExplanationReport]:
    """
    Generate pricing explanations for ALL products in a DataFrame.

    Adds the following columns to the DataFrame:
    - explanation_summary: One-line summary of the pricing rationale
    - explanation_bullets: Semicolon-separated bullet points
    - explanation_text: Full human-readable text block

    Args:
        df: DataFrame with at minimum 'current_price' and 'cost_price'.
            Also uses: optimal_price, competitor_price, demand_trend,
            sales_volume, inventory_level, price_elasticity.
        inplace: If True, modifies df in place; otherwise returns a copy.

    Returns:
        Tuple of (enriched DataFrame, ExplanationReport).
    """
    if not inplace:
        result = df.copy()
    else:
        result = df

    report = ExplanationReport()
    report.total_products = len(df)

    logger.info(f"batch_explanation: {len(df)} products")

    explanations: List[PriceExplanation] = []
    errors: int = 0
    error_ids: List[str] = []

    for idx, row in df.iterrows():
        try:
            product_data = row.to_dict()
            explanation = generate_explanation(product_data)
            explanations.append(explanation)
        except Exception as e:
            pid = str(row.get("product_id", idx))
            pname = str(row.get("product_name", ""))
            logger.warning(
                f"Explanation failed for '{pname}' ({pid}): {e}"
            )
            errors += 1
            error_ids.append(f"{pid} ({pname})")
            # Create a minimal error explanation
            explanations.append(
                PriceExplanation(
                    product_id=pid,
                    product_name=pname,
                    current_price=float(row.get("current_price", 0)),
                    optimal_price=float(row.get("optimal_price", row.get("current_price", 0))),
                    recommendation="Error",
                    summary=f"Explanation unavailable: {e}",
                    bullet_points=[f"✗ Error generating explanation: {e}"],
                )
            )

    report.explained = len(explanations) - errors
    report.errors = errors
    report.error_ids = error_ids

    # Write explanation columns to DataFrame
    result["explanation_summary"] = [e.summary for e in explanations]
    result["explanation_bullets"] = ["; ".join(e.bullet_points) for e in explanations]
    result["explanation_text"] = [e.to_text() for e in explanations]

    # Also write per-factor scores as structured columns
    for factor_name in ["demand", "inventory", "competitor", "margin", "cost"]:
        col_name = f"explanation_{factor_name}_direction"
        values: List[str] = []
        for e in explanations:
            match = [f for f in e.factors if f.label == factor_name or f.label.startswith(factor_name.replace("_", " "))]
            values.append(match[0].direction if match else "neutral")
        result[col_name] = values

    logger.info(
        f"batch_explanation complete: {report.explained} explained, "
        f"{report.errors} errors"
    )
    return result, report


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API
# ═══════════════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Pricing explainability engine.

    Generates human-readable explanations for pricing recommendations
    by analysing demand, inventory, competitor, margin, and cost factors.

    Each explanation includes:
    - One-line summary
    - Bullet points with icons (↑ ↓ ↗ ↘ →)
    - Per-factor detail with contribution estimates
    - Full text block

    Usage:
        engine = ExplainabilityEngine()
        df, report = engine.batch_explain(dataframe)
        print(df["explanation_summary"].iloc[0])

        # Single product
        expl = engine.explain_product(product_dict)
        print(expl.to_text())
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the explainability engine.

        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()
        self._last_report: Optional[ExplanationReport] = None

    @property
    def last_report(self) -> Optional[ExplanationReport]:
        """Get the report from the most recent batch explanation."""
        return self._last_report

    def explain_product(
        self,
        product_data: Dict[str, Any],
    ) -> PriceExplanation:
        """
        Generate a pricing explanation for a single product.

        Args:
            product_data: Dict with product fields.

        Returns:
            PriceExplanation with human-readable content.
        """
        return generate_explanation(product_data)

    def batch_explain(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
    ) -> Tuple[pd.DataFrame, ExplanationReport]:
        """
        Generate pricing explanations for all products.

        Args:
            df: Product DataFrame.
            inplace: If True, modifies df in place.

        Returns:
            Tuple of (enriched DataFrame, ExplanationReport).
        """
        result_df, report = batch_explanation(df, inplace=inplace)
        self._last_report = report
        return result_df, report

    def format_explanation(
        self,
        explanation: PriceExplanation,
        format: str = "text",
    ) -> str:
        """
        Format an explanation in a specific output format.

        Args:
            explanation: PriceExplanation object.
            format: Output format ('text', 'markdown', 'html').

        Returns:
            Formatted string.
        """
        if format == "text":
            return explanation.to_text()

        elif format == "markdown":
            lines = [
                f"### Pricing Recommendation: {explanation.product_name}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Current Price | ${explanation.current_price:.2f} |",
                f"| Optimal Price | ${explanation.optimal_price:.2f} |",
                f"| Change | {explanation.price_change_pct:+.1f}% |",
                f"| Recommendation | {explanation.recommendation} |",
                "",
                "**Reasons:**",
            ]
            for bullet in explanation.bullet_points:
                lines.append(f"- {bullet}")
            return "\n".join(lines)

        elif format == "html":
            html = f"<div class='pricing-explanation'>"
            html += f"<h3>Pricing Recommendation: {explanation.product_name}</h3>"
            html += f"<table><tr><th>Metric</th><th>Value</th></tr>"
            html += f"<tr><td>Current Price</td><td>${explanation.current_price:.2f}</td></tr>"
            html += f"<tr><td>Optimal Price</td><td>${explanation.optimal_price:.2f}</td></tr>"
            html += f"<tr><td>Change</td><td>{explanation.price_change_pct:+.1f}%</td></tr>"
            html += f"<tr><td>Recommendation</td><td>{explanation.recommendation}</td></tr>"
            html += f"</table>"
            html += f"<p><strong>Reasons:</strong></p><ul>"
            for bullet in explanation.bullet_points:
                html += f"<li>{bullet}</li>"
            html += "</ul></div>"
            return html

        else:
            return explanation.to_text()

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last batch explanation."""
        if self._last_report is None:
            return "No batch explanation performed yet."
        return self._last_report.summary()