"""
Scalable Pricing Engine Module

Enterprise-grade, rule-based pricing engine that computes optimal prices
for 1000+ products simultaneously using vectorized pandas operations.

Pricing Logic (per product):
  1. Cost-Plus Floor:    price >= cost × min_markup
  2. Competitor Anchor:  adjust toward competitor price with elasticity
  3. Demand Premium:     high demand → higher price tolerance
  4. Inventory Factor:   low inventory → slight price increase (scarcity)
  5. Margin Correction:  low margin → profitability correction
  6. Bounds:             clip between floor and ceiling

Output per product:
  - optimal_price
  - expected_revenue
  - expected_profit
  - margin_percentage
  - price_change_pct
  - recommendation (Increase / Decrease / Maintain)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.config import AppConfig
from utils.helpers import round_half_up, safe_divide

logger = logging.getLogger("ai_pricing.pricing_engine")


# ─── Constants ─────────────────────────────────────────────────────────────

# Default pricing parameters (can be overridden via config).
DEFAULT_PARAMS: Dict[str, float] = {
    # Cost-Plus
    "min_markup": 1.10,          # 10% minimum markup over cost
    "target_markup": 1.35,       # 35% target markup over cost
    "max_markup": 2.00,          # 100% maximum markup over cost

    # Competitor
    "competitor_weight": 0.35,   # How much to anchor toward competitor price
    "max_above_competitor": 1.20, # Max 20% above competitor
    "max_below_competitor": 0.70, # Max 30% below competitor

    # Demand
    "demand_premium_max": 0.15,  # Max 15% premium for high demand
    "demand_discount_max": 0.10, # Max 10% discount for low demand
    "high_demand_threshold": 0.7, # Demand_trend ≥ 0.7 = high
    "low_demand_threshold": 0.3,  # Demand_trend ≤ 0.3 = low

    # Inventory
    "inventory_scarcity_max": 0.08,   # Max 8% increase for low stock
    "inventory_discount_max": 0.05,    # Max 5% discount for excess stock
    "low_inventory_days": 15,          # Days-of-cover below this = scarce
    "excess_inventory_days": 90,       # Days-of-cover above this = excess

    # Margin Correction
    "min_acceptable_margin": 0.10,  # 10% minimum acceptable margin
    "margin_correction_strength": 0.20,  # How aggressively to fix low margin

    # Volume-Elasticity
    "elasticity_default": -1.5,     # Default elasticity if missing
    "volume_response_cap": 0.30,    # Max 30% volume change projection
}


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class PricingParams:
    """
    Configurable pricing parameters for the engine.

    Each parameter controls a specific aspect of the pricing logic.
    Can be initialised with defaults or overridden per call.
    """
    # Cost-Plus
    min_markup: float = DEFAULT_PARAMS["min_markup"]
    target_markup: float = DEFAULT_PARAMS["target_markup"]
    max_markup: float = DEFAULT_PARAMS["max_markup"]

    # Competitor
    competitor_weight: float = DEFAULT_PARAMS["competitor_weight"]
    max_above_competitor: float = DEFAULT_PARAMS["max_above_competitor"]
    max_below_competitor: float = DEFAULT_PARAMS["max_below_competitor"]

    # Demand
    demand_premium_max: float = DEFAULT_PARAMS["demand_premium_max"]
    demand_discount_max: float = DEFAULT_PARAMS["demand_discount_max"]
    high_demand_threshold: float = DEFAULT_PARAMS["high_demand_threshold"]
    low_demand_threshold: float = DEFAULT_PARAMS["low_demand_threshold"]

    # Inventory
    inventory_scarcity_max: float = DEFAULT_PARAMS["inventory_scarcity_max"]
    inventory_discount_max: float = DEFAULT_PARAMS["inventory_discount_max"]
    low_inventory_days: float = DEFAULT_PARAMS["low_inventory_days"]
    excess_inventory_days: float = DEFAULT_PARAMS["excess_inventory_days"]

    # Margin
    min_acceptable_margin: float = DEFAULT_PARAMS["min_acceptable_margin"]
    margin_correction_strength: float = DEFAULT_PARAMS["margin_correction_strength"]

    # Elasticity
    elasticity_default: float = DEFAULT_PARAMS["elasticity_default"]
    volume_response_cap: float = DEFAULT_PARAMS["volume_response_cap"]

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PricingReport:
    """
    Aggregate report for a pricing run.
    """
    total_products: int = 0
    products_with_optimal: int = 0
    products_with_errors: int = 0

    # Recommendation counts
    recommend_increase: int = 0
    recommend_decrease: int = 0
    recommend_maintain: int = 0

    # Revenue and profit impact
    total_current_revenue: float = 0.0
    total_expected_revenue: float = 0.0
    total_current_profit: float = 0.0
    total_expected_profit: float = 0.0
    avg_margin_current: float = 0.0
    avg_margin_optimal: float = 0.0
    avg_price_change_pct: float = 0.0

    # Error tracking
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "products": self.total_products,
            "optimal": self.products_with_optimal,
            "errors": self.products_with_errors,
            "recommend_increase": self.recommend_increase,
            "recommend_decrease": self.recommend_decrease,
            "recommend_maintain": self.recommend_maintain,
            "current_revenue": round(self.total_current_revenue, 2),
            "expected_revenue": round(self.total_expected_revenue, 2),
            "revenue_change": round(
                self.total_expected_revenue - self.total_current_revenue, 2
            ),
            "current_profit": round(self.total_current_profit, 2),
            "expected_profit": round(self.total_expected_profit, 2),
            "profit_change": round(
                self.total_expected_profit - self.total_current_profit, 2
            ),
            "avg_margin_current": round(self.avg_margin_current * 100, 2),
            "avg_margin_optimal": round(self.avg_margin_optimal * 100, 2),
            "avg_price_change_pct": round(self.avg_price_change_pct, 2),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        delta_str = (
            f"+${self.to_dict()['revenue_change']:,.0f}"
            if self.to_dict()["revenue_change"] >= 0
            else f"-${abs(self.to_dict()['revenue_change']):,.0f}"
        )
        profit_delta = self.to_dict()["profit_change"]
        profit_str = (
            f"+${profit_delta:,.0f}" if profit_delta >= 0 else f"-${abs(profit_delta):,.0f}"
        )

        lines = [
            f"  Products processed:     {self.total_products}",
            f"  Errors:                 {self.products_with_errors}",
            "",
            f"  Recommendations:",
            f"    Increase:  {self.recommend_increase}",
            f"    Decrease:  {self.recommend_decrease}",
            f"    Maintain:  {self.recommend_maintain}",
            "",
            f"  Revenue:  ${self.total_current_revenue:,.0f} → ${self.total_expected_revenue:,.0f}  ({delta_str})",
            f"  Profit:   ${self.total_current_profit:,.0f} → ${self.total_expected_profit:,.0f}  ({profit_str})",
            f"  Margin:   {self.avg_margin_current * 100:.1f}% → {self.avg_margin_optimal * 100:.1f}%",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CORE PRICING LOGIC  (Standalone Function)
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_pricing_inputs(df: pd.DataFrame, params: PricingParams) -> pd.DataFrame:
    """
    Validate and prepare the DataFrame for pricing calculations.

    Ensures all required columns exist, fills missing values with sensible
    defaults, and computes any intermediate columns needed.

    Args:
        df: Input DataFrame.
        params: Pricing parameters.

    Returns:
        DataFrame with guaranteed columns for pricing.

    Raises:
        ValueError: If critical columns are missing.
    """
    required = ["current_price", "cost_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df_out = df.copy()

    # Ensure numeric types
    for col in ["current_price", "cost_price"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    na_count = df_out[required].isna().any(axis=1).sum()
    if na_count > 0:
        logger.warning(f"{na_count} rows have NaN in required columns; they will produce NaN results.")

    # Competitor price: default to current_price * 1.05 if missing
    if "competitor_price" not in df_out.columns or df_out["competitor_price"].isna().all():
        df_out["competitor_price"] = df_out["current_price"] * 1.05
        logger.info("competitor_price not found; estimated as 105% of current_price.")
    else:
        df_out["competitor_price"] = pd.to_numeric(
            df_out["competitor_price"], errors="coerce"
        ).fillna(df_out["current_price"] * 1.05)

    # Sales volume / demand: default to 100 if missing
    if "sales_volume" not in df_out.columns:
        df_out["sales_volume"] = 100
    else:
        df_out["sales_volume"] = pd.to_numeric(
            df_out["sales_volume"], errors="coerce"
        ).fillna(100).clip(lower=0)

    # Inventory level: default to 500 if missing
    if "inventory_level" not in df_out.columns:
        df_out["inventory_level"] = 500
    else:
        df_out["inventory_level"] = pd.to_numeric(
            df_out["inventory_level"], errors="coerce"
        ).fillna(500).clip(lower=0)

    # Demand trend: default to 0.5 (neutral)
    if "demand_trend" not in df_out.columns:
        df_out["demand_trend"] = 0.5
    else:
        df_out["demand_trend"] = pd.to_numeric(
            df_out["demand_trend"], errors="coerce"
        ).fillna(0.5).clip(0.0, 1.0)

    # Price elasticity: default to -1.5
    if "price_elasticity" not in df_out.columns:
        df_out["price_elasticity"] = params.elasticity_default
    else:
        df_out["price_elasticity"] = pd.to_numeric(
            df_out["price_elasticity"], errors="coerce"
        ).fillna(params.elasticity_default).clip(-10.0, -0.01)

    return df_out


def _compute_cost_plus_price(df: pd.DataFrame, params: PricingParams) -> np.ndarray:
    """
    Compute cost-plus price floor: the minimum price to ensure target markup.

    cost_plus_price = cost_price × target_markup
    Also compute absolute floor: cost_price × min_markup

    Returns:
        Array of cost-plus recommended prices.
    """
    cost = df["cost_price"].values
    target_price = cost * params.target_markup
    floor_price = cost * params.min_markup
    return np.maximum(target_price, floor_price)


def _compute_competitor_adjustment(
    df: pd.DataFrame, cost_plus_price: np.ndarray, params: PricingParams
) -> np.ndarray:
    """
    Blend cost-plus price with competitor price using the competitor weight.

    If competitor is available:
      blended = (1 - w) × cost_plus + w × competitor
    Additionally clamp within [max_below_competitor, max_above_competitor]
    of the competitor price.

    Returns:
        Array of competitor-adjusted prices.
    """
    cost_plus = cost_plus_price
    comp = df["competitor_price"].values
    w = params.competitor_weight

    blended = (1.0 - w) * cost_plus + w * comp

    # Clamp relative to competitor
    upper_bound = comp * params.max_above_competitor
    lower_bound = comp * params.max_below_competitor
    clamped = np.clip(blended, lower_bound, upper_bound)

    return clamped


def _compute_demand_premium(
    df: pd.DataFrame, params: PricingParams
) -> np.ndarray:
    """
    Compute a demand-based multiplier.

    High demand (≥ threshold) → positive premium (up to demand_premium_max)
    Low demand (≤ threshold) → negative discount (down to demand_discount_max)
    Neutral → 0

    Returns:
        Array of demand adjustment factors (1.0 + premium).
    """
    demand = df["demand_trend"].values

    # Premium for high demand
    high_mask = demand >= params.high_demand_threshold
    high_scale = (demand - params.high_demand_threshold) / (1.0 - params.high_demand_threshold)
    premium = np.where(high_mask, high_scale * params.demand_premium_max, 0.0)

    # Discount for low demand
    low_mask = demand <= params.low_demand_threshold
    low_scale = (params.low_demand_threshold - demand) / params.low_demand_threshold
    discount = np.where(low_mask, low_scale * params.demand_discount_max, 0.0)

    adjustment = premium - discount
    return 1.0 + adjustment


def _compute_inventory_adjustment(
    df: pd.DataFrame, params: PricingParams
) -> np.ndarray:
    """
    Compute an inventory-based adjustment factor.

    Low days-of-cover (< low_inventory_days) → scarcity premium
    High days-of-cover (> excess_inventory_days) → excess discount
    Healthy → no adjustment

    If days_of_cover not available, uses inventory_level / sales_volume as proxy.

    Returns:
        Array of inventory adjustment factors (1.0 + adjustment).
    """
    if "days_of_cover" in df.columns:
        doc = df["days_of_cover"].values
    else:
        vol = df["sales_volume"].values
        inv = df["inventory_level"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            doc = np.where(vol > 0, (inv / vol) * 30, 999.0)

    # Scarcity premium
    scarce_mask = doc < params.low_inventory_days
    scarce_scale = np.where(
        scarce_mask,
        (params.low_inventory_days - doc) / params.low_inventory_days,
        0.0,
    )
    scarcity_premium = np.clip(scarce_scale, 0.0, 1.0) * params.inventory_scarcity_max

    # Excess discount
    excess_mask = doc > params.excess_inventory_days
    excess_scale = np.where(
        excess_mask,
        (doc - params.excess_inventory_days) / params.excess_inventory_days,
        0.0,
    )
    excess_discount = np.clip(excess_scale, 0.0, 1.0) * params.inventory_discount_max

    adjustment = scarcity_premium - excess_discount
    return 1.0 + adjustment


def _compute_margin_correction(
    df: pd.DataFrame, base_price: np.ndarray, params: PricingParams
) -> np.ndarray:
    """
    Apply a margin correction if the price results in below-minimum margin.

    If (base_price - cost) / base_price < min_acceptable_margin, raise price.

    Returns:
        Array of margin-corrected prices.
    """
    cost = df["cost_price"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(base_price > 0, (base_price - cost) / base_price, 0.0)

    low_margin_mask = margin < params.min_acceptable_margin

    # Price needed to achieve min_acceptable_margin
    # margin = (price - cost) / price  =>  price = cost / (1 - margin)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction_price = np.where(
            low_margin_mask,
            cost / (1.0 - params.min_acceptable_margin),
            base_price,
        )

    # Blend: partial correction to avoid abrupt jumps
    corrected = np.where(
        low_margin_mask,
        (1.0 - params.margin_correction_strength) * base_price
        + params.margin_correction_strength * correction_price,
        base_price,
    )

    return corrected


def _apply_bounds(
    price: np.ndarray, df: pd.DataFrame, params: PricingParams
) -> np.ndarray:
    """
    Apply absolute bounds to the optimal price.

    Floor: cost_price × min_markup
    Ceiling: cost_price × max_markup

    Returns:
        Bounded price array.
    """
    cost = df["cost_price"].values
    floor = cost * params.min_markup
    ceiling = cost * params.max_markup
    return np.clip(price, floor, ceiling)


def _compute_projections(
    optimal_price: np.ndarray,
    df: pd.DataFrame,
    params: PricingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute expected revenue, profit, margin, and volume projections.

    Volume response is estimated using price elasticity:
      volume_change_pct = elasticity × price_change_pct
      new_volume = current_volume × (1 + volume_change_pct)

    Revenue = optimal_price × new_volume
    Profit = (optimal_price - cost) × new_volume
    Margin = (optimal_price - cost) / optimal_price

    Returns:
        Tuple of (expected_revenue, expected_profit, margin_pct, new_volume).
    """
    current_price = df["current_price"].values
    cost = df["cost_price"].values
    current_volume = df["sales_volume"].values
    elasticity = df["price_elasticity"].values

    # Price change percentage
    with np.errstate(divide="ignore", invalid="ignore"):
        price_change_pct = np.where(
            current_price > 0, (optimal_price - current_price) / current_price, 0.0
        )

    # Volume response (capped to avoid unrealistic projections)
    raw_volume_change = elasticity * price_change_pct
    volume_change = np.clip(raw_volume_change, -params.volume_response_cap, params.volume_response_cap)
    new_volume = current_volume * (1.0 + volume_change)
    new_volume = np.maximum(new_volume, 0)

    # Revenue and profit
    expected_revenue = optimal_price * new_volume
    expected_profit = (optimal_price - cost) * new_volume

    # Margin
    with np.errstate(divide="ignore", invalid="ignore"):
        margin_pct = np.where(
            optimal_price > 0, (optimal_price - cost) / optimal_price, 0.0
        )

    return expected_revenue, expected_profit, margin_pct, new_volume


def compute_optimal_prices(
    df: pd.DataFrame,
    params: Optional[PricingParams] = None,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, PricingReport]:
    """
    Main pricing engine function — compute optimal prices for ALL products
    using fully vectorised pandas/numpy operations.

    Pipeline (applied per product):
      1. Cost-Plus Floor
      2. Competitor Blending
      3. Demand Premium/Discount
      4. Inventory Scarcity/Excess
      5. Margin Correction
      6. Absolute Bounds

    Then projects expected revenue, profit, and margin.

    Args:
        df: DataFrame with at least 'current_price' and 'cost_price'.
            Optional columns: competitor_price, sales_volume, inventory_level,
            demand_trend, price_elasticity.
        params: Pricing parameters (uses defaults if None).
        inplace: If True, modifies df in place; otherwise returns a copy.

    Returns:
        Tuple of:
        - Enriched DataFrame with columns:
            optimal_price, expected_revenue, expected_profit,
            margin_percentage, price_change_pct, recommendation,
            volume_projected
        - PricingReport with aggregate statistics.

    Raises:
        ValueError: If required columns are missing.
    """
    if params is None:
        params = PricingParams()

    logger.info(
        f"compute_optimal_prices: {len(df)} products, "
        f"params={params.to_dict()}"
    )

    report = PricingReport()
    report.total_products = len(df)

    if not inplace:
        result = df.copy()
    else:
        result = df

    try:
        # Step 0: Prepare / validate inputs
        data = _prepare_pricing_inputs(result, params)

        # Step 1: Cost-Plus Floor
        cost_plus = _compute_cost_plus_price(data, params)

        # Step 2: Competitor Blending
        comp_adjusted = _compute_competitor_adjustment(data, cost_plus, params)

        # Step 3: Demand Premium / Discount
        demand_factor = _compute_demand_premium(data, params)
        demand_adjusted = comp_adjusted * demand_factor

        # Step 4: Inventory Adjustment
        inventory_factor = _compute_inventory_adjustment(data, params)
        inv_adjusted = demand_adjusted * inventory_factor

        # Step 5: Margin Correction
        margin_corrected = _compute_margin_correction(data, inv_adjusted, params)

        # Step 6: Absolute Bounds
        optimal_price = _apply_bounds(margin_corrected, data, params)

        # Round to 2 decimal places
        optimal_price = np.round(optimal_price, 2)

        # Step 7: Compute projections
        expected_revenue, expected_profit, margin_pct, new_volume = (
            _compute_projections(optimal_price, data, params)
        )

        # Step 8: Current values for reporting
        current_price = data["current_price"].values
        cost = data["cost_price"].values
        current_volume = data["sales_volume"].values
        current_revenue = current_price * current_volume
        current_profit = (current_price - cost) * current_volume

        # Price change %
        with np.errstate(divide="ignore", invalid="ignore"):
            price_change_pct = np.where(
                current_price > 0,
                (optimal_price - current_price) / current_price,
                0.0,
            )

        # Recommendation
        recommendations = np.where(
            price_change_pct > 0.02, "Increase",
            np.where(price_change_pct < -0.02, "Decrease", "Maintain")
        )

        # Write results to DataFrame
        result["optimal_price"] = optimal_price
        result["expected_revenue"] = np.round(expected_revenue, 2)
        result["expected_profit"] = np.round(expected_profit, 2)
        result["expected_margin"] = np.round(margin_pct * 100, 2)
        result["margin_percentage"] = np.round(margin_pct * 100, 2)  # alias
        result["price_change_pct"] = np.round(price_change_pct * 100, 2)
        result["recommendation"] = recommendations
        result["volume_projected"] = np.round(new_volume, 0)

        # Populate report
        report.total_products = len(result)
        report.products_with_optimal = int(np.isfinite(optimal_price).sum())
        report.products_with_errors = int((~np.isfinite(optimal_price)).sum())
        report.recommend_increase = int((recommendations == "Increase").sum())
        report.recommend_decrease = int((recommendations == "Decrease").sum())
        report.recommend_maintain = int((recommendations == "Maintain").sum())
        report.total_current_revenue = float(np.nansum(current_revenue))
        report.total_expected_revenue = float(np.nansum(expected_revenue))
        report.total_current_profit = float(np.nansum(current_profit))
        report.total_expected_profit = float(np.nansum(expected_profit))
        report.avg_margin_current = float(
            np.nanmean(
                np.where(
                    current_price > 0,
                    (current_price - cost) / current_price,
                    np.nan,
                )
            )
        )
        report.avg_margin_optimal = float(np.nanmean(margin_pct))
        report.avg_price_change_pct = float(np.nanmean(price_change_pct)) * 100

        logger.info(
            f"Pricing complete: {report.products_with_optimal} optimal prices, "
            f"rev ${report.total_current_revenue:,.0f} → ${report.total_expected_revenue:,.0f}, "
            f"margin {report.avg_margin_current*100:.1f}% → {report.avg_margin_optimal*100:.1f}%"
        )

    except Exception as e:
        logger.error(f"Pricing engine error: {e}", exc_info=True)
        report.errors.append(str(e))
        # Still return the DataFrame with NaN for pricing columns
        for col in [
            "optimal_price", "expected_revenue", "expected_profit",
            "margin_percentage", "price_change_pct", "volume_projected",
        ]:
            result[col] = np.nan
        result["recommendation"] = "Error"

    return result, report


# ─── Standalone Batch Pricing ─────────────────────────────────────────────

def batch_optimize_prices(
    df: pd.DataFrame,
    params: Optional[PricingParams] = None,
    chunk_size: int = 5000,
) -> Tuple[pd.DataFrame, PricingReport]:
    """
    Batch-optimize prices for 1000+ products with chunked processing.

    For extremely large datasets, processes in chunks to manage memory.
    Each chunk is processed independently with compute_optimal_prices.

    Args:
        df: Product DataFrame.
        params: Pricing parameters.
        chunk_size: Number of products per chunk (default 5000).

    Returns:
        Tuple of (enriched DataFrame, merged PricingReport).
    """
    total = len(df)
    logger.info(f"Batch optimizing {total} products (chunk_size={chunk_size})")

    if total <= chunk_size:
        return compute_optimal_prices(df, params=params, inplace=False)

    # Process in chunks
    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, total, chunk_size)]
    chunk_results: List[pd.DataFrame] = []
    combined_report = PricingReport()
    combined_report.total_products = total

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} products)")
        chunk_result, chunk_report = compute_optimal_prices(
            chunk, params=params, inplace=True
        )
        chunk_results.append(chunk_result)

        # Aggregate report
        combined_report.products_with_optimal += chunk_report.products_with_optimal
        combined_report.products_with_errors += chunk_report.products_with_errors
        combined_report.recommend_increase += chunk_report.recommend_increase
        combined_report.recommend_decrease += chunk_report.recommend_decrease
        combined_report.recommend_maintain += chunk_report.recommend_maintain
        combined_report.total_current_revenue += chunk_report.total_current_revenue
        combined_report.total_expected_revenue += chunk_report.total_expected_revenue
        combined_report.total_current_profit += chunk_report.total_current_profit
        combined_report.total_expected_profit += chunk_report.total_expected_profit
        combined_report.errors.extend(chunk_report.errors)

    # Compute weighted averages for margin and price change
    if total > 0:
        combined_report.avg_margin_current = (
            combined_report.total_current_profit / combined_report.total_current_revenue
            if combined_report.total_current_revenue > 0
            else 0.0
        )
        combined_report.avg_margin_optimal = (
            combined_report.total_expected_profit / combined_report.total_expected_revenue
            if combined_report.total_expected_revenue > 0
            else 0.0
        )

    result_df = pd.concat(chunk_results, ignore_index=True)
    logger.info(
        f"Batch complete: {combined_report.products_with_optimal} optimal, "
        f"{combined_report.products_with_errors} errors"
    )
    return result_df, combined_report


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API  (backward-compatible wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class PricingEngine:
    """
    Enterprise-grade pricing engine using rule-based business logic.

    Computes optimal prices for 1000+ products simultaneously using
    fully vectorised operations. No ML training required — the engine
    applies configurable business rules around cost, competitor pricing,
    demand, inventory, and margin targets.

    Pipeline:
      1. Cost-Plus Floor         — price ≥ cost × min_markup
      2. Competitor Blending     — anchor toward competitor price
      3. Demand Premium          — high demand → higher price
      4. Inventory Adjustment    — scarcity → increase, excess → discount
      5. Margin Correction       — low margin → profitability fix
      6. Absolute Bounds         — clip between floor and ceiling
      7. Projections             — revenue, profit, margin, volume

    Usage:
        engine = PricingEngine()
        df, report = engine.optimize_all(dataframe)
        print(report.summary())
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        params: Optional[PricingParams] = None,
    ) -> None:
        """
        Initialize the pricing engine.

        Args:
            config: Application configuration (for thresholds).
            params: Pricing parameters (uses defaults if None).
        """
        self.config = config or AppConfig()
        self.params = params or PricingParams()
        self._last_report: Optional[PricingReport] = None

    @property
    def last_report(self) -> Optional[PricingReport]:
        """Get the report from the most recent pricing run."""
        return self._last_report

    @last_report.setter
    def last_report(self, report: PricingReport) -> None:
        """Set the last report (used for testing)."""
        self._last_report = report

    def optimize_all(
        self,
        df: pd.DataFrame,
        inplace: bool = False,
        **param_overrides: Any,
    ) -> Tuple[pd.DataFrame, PricingReport]:
        """
        Compute optimal prices for ALL products in the DataFrame.

        This is the primary entry point. Calls compute_optimal_prices()
        with the configured parameters.

        Args:
            df: Product DataFrame with at least 'current_price' and 'cost_price'.
            inplace: If True, modifies df in place.
            **param_overrides: Override individual pricing parameters
                               (e.g., min_markup=1.15, competitor_weight=0.4).

        Returns:
            Tuple of (enriched DataFrame, PricingReport).

        Raises:
            ValueError: If required columns are missing.
        """
        # Apply parameter overrides
        if param_overrides:
            params_dict = self.params.to_dict()
            params_dict.update(param_overrides)
            params = PricingParams(**params_dict)
        else:
            params = self.params

        result_df, report = compute_optimal_prices(
            df, params=params, inplace=inplace
        )
        self._last_report = report
        return result_df, report

    def optimize_single(
        self,
        current_price: float,
        cost_price: float,
        competitor_price: Optional[float] = None,
        sales_volume: Optional[float] = None,
        inventory_level: Optional[float] = None,
        demand_trend: Optional[float] = None,
        price_elasticity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute optimal price for a single product.

        Convenience method that wraps a single row into a DataFrame
        and calls optimize_all.

        Args:
            current_price: Current selling price.
            cost_price: Unit cost.
            competitor_price: Average competitor price.
            sales_volume: Units sold per period.
            inventory_level: Current stock level.
            demand_trend: Demand signal (0-1).
            price_elasticity: Price sensitivity coefficient.

        Returns:
            Dict with pricing results for the product.
        """
        row = {
            "current_price": current_price,
            "cost_price": cost_price,
        }
        if competitor_price is not None:
            row["competitor_price"] = competitor_price
        if sales_volume is not None:
            row["sales_volume"] = sales_volume
        if inventory_level is not None:
            row["inventory_level"] = inventory_level
        if demand_trend is not None:
            row["demand_trend"] = demand_trend
        if price_elasticity is not None:
            row["price_elasticity"] = price_elasticity

        df = pd.DataFrame([row])
        result_df, _ = self.optimize_all(df, inplace=False)

        if result_df.empty:
            return {"error": "No result produced"}

        row_out = result_df.iloc[0].to_dict()
        # Drop NaN values for cleanliness
        return {k: v for k, v in row_out.items() if not (isinstance(v, float) and np.isnan(v))}

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last pricing run."""
        if self._last_report is None:
            return "No pricing run performed yet."
        return self._last_report.summary()
