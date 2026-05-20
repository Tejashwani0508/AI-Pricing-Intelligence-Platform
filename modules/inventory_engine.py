"""
Inventory Pricing Optimization Engine

Analyses inventory levels alongside demand signals to generate pricing
recommendations. The core logic is a 2Ã—2 decision matrix:

                | High Demand      | Low Demand
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Low Stock   | â†‘ PRICE INCREASE | â˜… Monitor
                | (scarcity premium)| (natural run-down)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    High Stock  | â˜… Maintain      | â†“ DISCOUNT
                | (healthy)       | (clearance)

Output per product:
  - inventory_score      (0-100, higher = healthier)
  - inventory_risk       (LOW / MEDIUM / HIGH / CRITICAL)
  - inventory_action     (pricing action: "increase" / "discount" / "maintain" / "monitor")
  - days_of_cover        (estimated stock coverage in days)
  - stock_status         (Healthy / Low Stock / Overstocked)
  - price_adjustment_pct (recommended % change in price)
  - reorder_point        (units at which to reorder)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import AppConfig
logger = logging.getLogger("ai_pricing.inventory_engine")


# â”€â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Days of cover thresholds
LOW_STOCK_DAYS: float = 15.0        # Below this â†’ low stock
HEALTHY_MIN_DAYS: float = 15.0      # Minimum healthy cover
HEALTHY_MAX_DAYS: float = 60.0      # Maximum healthy cover
OVERSTOCK_DAYS: float = 90.0        # Above this â†’ overstocked


# Demand thresholds (0-1 scale)
HIGH_DEMAND_THRESHOLD: float = 0.7
LOW_DEMAND_THRESHOLD: float = 0.3

# Pricing adjustments (percentage change)
SCARCITY_PREMIUM_MAX: float = 0.12      # Max 12% increase for scarcity
DISCOUNT_MAX: float = 0.20              # Max 20% discount for clearance
MAINTAIN_THRESHOLD: float = 0.02        # Â±2% â†’ "maintain"

# Inventory score thresholds
SCORE_HIGH: float = 70.0
SCORE_MEDIUM: float = 40.0

# Reorder parameters
DEFAULT_LEAD_TIME_DAYS: int = 14
SAFETY_STOCK_DAYS: int = 7

# Markdown / clearance parameters
OVERSTOCK_DISCOUNT_DAYS: float = 90.0  # Start discounting after this many days of cover


# â”€â”€â”€ Data Classes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class InventoryAnalysis:
    """
    Complete inventory + pricing analysis for a single product.
    """
    product_id: str = ""
    product_name: str = ""
    inventory_level: float = 0.0
    sales_volume: float = 0.0
    days_of_cover: float = 0.0
    inventory_score: float = 50.0
    inventory_risk: str = "MEDIUM"
    stock_status: str = "Healthy"
    inventory_action: str = "maintain"
    price_adjustment_pct: float = 0.0
    demand_trend: float = 0.5
    reorder_point: float = 0.0
    safety_stock: float = 0.0
    recommended_order_qty: float = 0.0
    is_understocked: bool = False
    is_overstocked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "inventory_level": round(self.inventory_level, 0),
            "sales_volume": round(self.sales_volume, 0),
            "days_of_cover": round(self.days_of_cover, 1),
            "inventory_score": round(self.inventory_score, 1),
            "inventory_risk": self.inventory_risk,
            "stock_status": self.stock_status,
            "inventory_action": self.inventory_action,
            "price_adjustment_pct": round(self.price_adjustment_pct, 1),
            "demand_trend": round(self.demand_trend, 2),
            "reorder_point": round(self.reorder_point, 0),
            "safety_stock": round(self.safety_stock, 0),
        }


@dataclass
class InventoryReport:
    """
    Aggregate report for a batch inventory analysis.
    """
    total_products: int = 0
    understocked_count: int = 0
    overstocked_count: int = 0
    healthy_count: int = 0

    actions_increase: int = 0
    actions_discount: int = 0
    actions_maintain: int = 0
    actions_monitor: int = 0

    avg_days_of_cover: float = 0.0
    avg_inventory_score: float = 0.0

    risk_distribution: Dict[str, int] = field(default_factory=dict)
    stock_status_distribution: Dict[str, int] = field(default_factory=dict)

    total_stock_value: float = 0.0
    products_needing_reorder: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_products,
            "understocked": self.understocked_count,
            "overstocked": self.overstocked_count,
            "healthy": self.healthy_count,
            "increase_price": self.actions_increase,
            "discount": self.actions_discount,
            "maintain": self.actions_maintain,
            "monitor": self.actions_monitor,
            "avg_days_cover": round(self.avg_days_of_cover, 1),
            "avg_score": round(self.avg_inventory_score, 1),
            "needs_reorder": self.products_needing_reorder,
            "total_stock_value": round(self.total_stock_value, 2),
        }

    def summary(self) -> str:
        return (
            f"  Products:           {self.total_products}\n"
            f"  Healthy:            {self.healthy_count}\n"
            f"  Understocked:       {self.understocked_count}\n"
            f"  Overstocked:        {self.overstocked_count}\n"
            f"  Actions:\n"
            f"    Increase price:   {self.actions_increase}\n"
            f"    Discount:         {self.actions_discount}\n"
            f"    Maintain:         {self.actions_maintain}\n"
            f"    Monitor:          {self.actions_monitor}\n"
            f"  Avg days of cover:  {self.avg_days_of_cover:.0f}\n"
            f"  Avg score:          {self.avg_inventory_score:.1f}"
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CORE INVENTORY METRICS (vectorised)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _compute_days_of_cover(
    inventory_level: np.ndarray,
    sales_volume: np.ndarray,
) -> np.ndarray:
    """
    Compute estimated days of cover for each product.

    days_of_cover = inventory_level / (sales_volume / 30)

    Args:
        inventory_level: Current stock units.
        sales_volume: Monthly sales units.

    Returns:
        Array of days of cover (capped at 999).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_sales = sales_volume / 30.0
        doc = np.where(
            daily_sales > 0,
            inventory_level / daily_sales,
            np.where(inventory_level > 0, 999.0, 0.0),
        )
    return np.clip(doc, 0, 999)


def _compute_inventory_score(
    days_of_cover: np.ndarray,
    inventory_level: np.ndarray,
    sales_volume: np.ndarray,
) -> np.ndarray:
    """
    Compute inventory health score (0-100).

    Components:
    1. Days of Cover (0-60): 15-60 days â†’ high score
    2. Turnover (0-30): 2-8x monthly â†’ high score
    3. Stockout avoidance (0-10): > 0 units â†’ full points

    Args:
        days_of_cover: Days of cover array.
        inventory_level: Current stock.
        sales_volume: Monthly sales.

    Returns:
        Array of scores (0-100).
    """
    n = len(days_of_cover)
    scores = np.zeros(n, dtype=float)

    # 1. Days of Cover Component (0-60)
    doc_score = np.where(
        (days_of_cover >= HEALTHY_MIN_DAYS) & (days_of_cover <= HEALTHY_MAX_DAYS),
        60.0,
        np.where(
            (days_of_cover >= LOW_STOCK_DAYS) & (days_of_cover <= OVERSTOCK_DAYS),
            40.0,
            np.where(
                days_of_cover < LOW_STOCK_DAYS,
                np.clip(days_of_cover / LOW_STOCK_DAYS * 30, 0, 30),
                # Overstocked (high days of cover)
                np.clip(
                    (1.0 - (days_of_cover - OVERSTOCK_DAYS) / (365.0 - OVERSTOCK_DAYS)) * 30,
                    5, 30,
                ),
            ),
        ),
    )

    # 2. Turnover Component (0-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = np.where(inventory_level > 0, sales_volume / inventory_level, 0.0)

    turnover_score = np.where(
        (turnover >= 2.0) & (turnover <= 8.0),
        30.0,
        np.where(
            turnover >= 1.0,
            20.0,
            np.where(
                (turnover > 0) & (turnover < 1.0),
                turnover * 20,
                0.0,
            ),
        ),
    )

    # 3. Stockout Avoidance (0-10)
    stockout_score = np.where(
        inventory_level > 0,
        np.clip(np.log10(inventory_level + 1) / 3 * 10, 0, 10),
        0.0,
    )

    scores = doc_score + turnover_score + stockout_score
    return np.clip(scores, 0, 100)


def _classify_risk(score: np.ndarray) -> np.ndarray:
    """
    Classify inventory risk level from score.

    Args:
        score: Inventory score array (0-100).

    Returns:
        Array of risk strings: LOW / MEDIUM / HIGH / CRITICAL.
    """
    return np.select(
        [score >= SCORE_HIGH, score >= SCORE_MEDIUM, score >= 20.0],
        ["LOW", "MEDIUM", "HIGH"],
        default="CRITICAL",
    )


def _classify_stock_status(days_of_cover: np.ndarray) -> np.ndarray:
    """
    Classify stock status based on days of cover.

    Args:
        days_of_cover: Days of cover array.

    Returns:
        Array of status strings.
    """
    return np.select(
        [
            days_of_cover == 0,
            (days_of_cover > 0) & (days_of_cover < LOW_STOCK_DAYS),
            (days_of_cover >= LOW_STOCK_DAYS) & (days_of_cover <= HEALTHY_MAX_DAYS),
            days_of_cover > HEALTHY_MAX_DAYS,
        ],
        ["Out of Stock", "Low Stock", "Healthy", "Overstocked"],
        default="Overstocked",
    )


def _detect_understocked(
    days_of_cover: np.ndarray, inventory_level: np.ndarray,
) -> np.ndarray:
    """Detect products that are understocked (days_of_cover < LOW_STOCK_DAYS)."""
    return days_of_cover < LOW_STOCK_DAYS


def _detect_overstocked(
    days_of_cover: np.ndarray, inventory_level: np.ndarray,
) -> np.ndarray:
    """Detect products that are overstocked (days_of_cover > HEALTHY_MAX_DAYS)."""
    return days_of_cover > HEALTHY_MAX_DAYS


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PRICING LOGIC â€” Inventory Ã— Demand Matrix
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _compute_price_adjustment(
    days_of_cover: np.ndarray,
    demand_trend: np.ndarray,
    current_price: np.ndarray,
    cost_price: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute price adjustment based on the inventory Ã— demand matrix.

    Matrix:
                    | High Demand (â‰¥0.7) | Medium (0.3-0.7) | Low Demand (â‰¤0.3)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Low Stock       | â†‘ +12% scarcity   â”‚ +5%              â”‚ 0% (run down)
    (<15d)          | premium           â”‚ slight increase  â”‚
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Healthy         â”‚ 0% maintain       â”‚ 0% maintain      â”‚ 0% maintain
    (15-60d)        â”‚                   â”‚                  â”‚
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Overstocked     â”‚ 0% maintain       â”‚ -5% slight       â”‚ -10% discount
    (60-180d)       â”‚                   â”‚ discount         â”‚
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Excess Stock    â”‚ -5% light         â”‚ -10% discount    â”‚ -20% clearance
    (>180d)         â”‚ discount          â”‚                  â”‚

    Also considers cost floor: price cannot go below cost Ã— 1.05.

    Args:
        days_of_cover: Days of cover array.
        demand_trend: Demand signal (0-1).
        current_price: Current selling price.
        cost_price: Unit cost.

    Returns:
        Tuple of (adjustment_pct, action, reason_index).
        adjustment_pct: Recommended % price change (e.g. 0.12 = +12%).
        action: "increase" / "discount" / "maintain" / "monitor".
    """
    n = len(days_of_cover)
    adjustment = np.zeros(n, dtype=float)
    action = np.full(n, "maintain", dtype=object)
    # Intermediate arrays for decision
    is_understocked = days_of_cover < LOW_STOCK_DAYS
    is_healthy = (days_of_cover >= LOW_STOCK_DAYS) & (days_of_cover <= HEALTHY_MAX_DAYS)
    is_overstocked = days_of_cover > HEALTHY_MAX_DAYS
    is_high_demand = demand_trend >= HIGH_DEMAND_THRESHOLD
    is_low_demand = demand_trend <= LOW_DEMAND_THRESHOLD
    is_medium_demand = ~is_high_demand & ~is_low_demand

    # â”€â”€â”€ Understocked â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Low stock + High demand â†’ scarcity premium (increase price)
    mask = is_understocked & is_high_demand
    adjustment[mask] = SCARCITY_PREMIUM_MAX
    action[mask] = "increase"

    # Low stock + Medium demand â†’ slight increase
    mask = is_understocked & is_medium_demand
    adjustment[mask] = 0.05
    action[mask] = "increase"

    # Low stock + Low demand â†’ no change (let it run down naturally)
    mask = is_understocked & is_low_demand
    adjustment[mask] = 0.0
    action[mask] = "monitor"

    # â”€â”€â”€ Healthy stock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mask = is_healthy
    adjustment[mask] = 0.0
    action[mask] = "maintain"

    # â”€â”€â”€ Overstocked â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Overstocked + High demand â†’ maintain (demand will clear it)
    mask = is_overstocked & is_high_demand
    adjustment[mask] = 0.0
    action[mask] = "maintain"

    # Overstocked + Medium demand â†’ slight discount
    mask = is_overstocked & is_medium_demand
    adjustment[mask] = -0.05
    action[mask] = "discount"

    # Overstocked + Low demand â†’ discount
    mask = is_overstocked & is_low_demand
    adjustment[mask] = -DISCOUNT_MAX
    action[mask] = "discount"

    # â”€â”€â”€ Excess stock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Excess + High demand â†’ light discount
    # â”€â”€â”€ Apply cost floor constraint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Ensure price doesn't go below cost Ã— 1.05
    with np.errstate(divide="ignore", invalid="ignore"):
        max_discount = np.where(
            current_price > 0,
            (current_price - cost_price * 1.05) / current_price,
            0.0,
        )
    # Clip adjustment so it doesn't exceed max_discount
    adjustment = np.where(
        adjustment < -max_discount,
        -max_discount,
        adjustment,
    )
    # If cost floor prevents any meaningful discount, change action
    floor_blocked = (adjustment > -0.01) & (action == "discount")
    adjustment[floor_blocked] = 0.0
    action[floor_blocked] = "maintain"

    return adjustment, action


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# REORDER LOGIC
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _compute_reorder_metrics(
    days_of_cover: np.ndarray,
    inventory_level: np.ndarray,
    sales_volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reorder point, safety stock, and recommended order quantity.

    Args:
        days_of_cover: Days of cover.
        inventory_level: Current stock.
        sales_volume: Monthly sales.

    Returns:
        Tuple of (reorder_point, safety_stock, recommended_order_qty).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_sales = sales_volume / 30.0
        safety_stock = daily_sales * SAFETY_STOCK_DAYS
        reorder_point = daily_sales * DEFAULT_LEAD_TIME_DAYS + safety_stock

    # Recommended order: enough for 30 days minus current inventory
    recommended_order = np.maximum(0, sales_volume - inventory_level)

    safety_stock = np.maximum(0, safety_stock)
    reorder_point = np.maximum(0, reorder_point)

    return np.round(reorder_point, 0), np.round(safety_stock, 0), np.round(recommended_order, 0)


def analyze_inventory(
    df: pd.DataFrame,
    inventory_col: str = "inventory_level",
    sales_col: str = "sales_volume",
    demand_col: str = "demand_trend",
    price_col: str = "current_price",
    cost_col: str = "cost_price",
    category_col: str = "category",
    inplace: bool = False,
) -> Tuple[pd.DataFrame, InventoryReport]:
    """
    Perform comprehensive inventory + pricing optimisation analysis.

    Computes:
    - Days of cover
    - Inventory health score (0-100)
    - Inventory risk level
    - Stock status
    - Inventory Ã— Demand pricing decisions (increase/discount/maintain/monitor)
    - Price adjustment percentage
    - Reorder point & safety stock

    - Stock value

    Args:
        df: DataFrame with inventory, sales, demand, and pricing data.
        inventory_col: Column with inventory level.
        sales_col: Column with sales volume.
        demand_col: Column with demand trend (0-1).
        price_col: Column with current price.
        cost_col: Column with cost price.
        category_col: Column with category.
        inplace: If True, modifies df in place.

    Returns:
        Tuple of (enriched DataFrame, InventoryReport).

    Raises:
        ValueError: If required columns are missing.
    """
    if not inplace:
        result = df.copy()
    else:
        result = df

    report = InventoryReport()
    report.total_products = len(df)

    # â”€â”€â”€ Validate required columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    required = [inventory_col, sales_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    n = len(df)
    logger.info(f"analyze_inventory: {n} products")

    # â”€â”€â”€ Extract arrays with defaults â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    inventory_level = pd.to_numeric(df[inventory_col], errors="coerce").fillna(0).clip(0).values
    sales_volume = pd.to_numeric(df[sales_col], errors="coerce").fillna(0).clip(0).values
    demand_trend = (
        pd.to_numeric(df[demand_col], errors="coerce").fillna(0.5).clip(0, 1).values
        if demand_col in df.columns
        else np.full(n, 0.5)
    )
    current_price = (
        pd.to_numeric(df[price_col], errors="coerce").fillna(0).values
        if price_col in df.columns
        else np.full(n, 100.0)
    )
    cost_price = (
        pd.to_numeric(df[cost_col], errors="coerce")
        .fillna(pd.Series(current_price * 0.6, index=df.index))
        .values
        if cost_col in df.columns
        else current_price * 0.6
    )

    # â”€â”€â”€ Compute core metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    days_of_cover = _compute_days_of_cover(inventory_level, sales_volume)
    inventory_score = _compute_inventory_score(days_of_cover, inventory_level, sales_volume)
    risk_level = _classify_risk(inventory_score)
    stock_status = _classify_stock_status(days_of_cover)

    # Understocked / overstocked flags
    understocked = _detect_understocked(days_of_cover, inventory_level)
    overstocked = _detect_overstocked(days_of_cover, inventory_level)

    # â”€â”€â”€ Pricing decisions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    price_adjustment, inventory_action = _compute_price_adjustment(
        days_of_cover, demand_trend, current_price, cost_price,
    )
    price_adjustment_pct = np.round(price_adjustment * 100, 1)

    # â”€â”€â”€ Reorder metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reorder_point, safety_stock, recommended_order = _compute_reorder_metrics(
        days_of_cover, inventory_level, sales_volume,
    )

    # â”€â”€â”€ Stock value â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    stock_value = inventory_level * cost_price


    # â”€â”€â”€ Write results to DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    result["days_of_cover"] = np.round(days_of_cover, 1)
    result["inventory_score"] = np.round(inventory_score, 1)
    result["inventory_risk"] = risk_level
    result["stock_status"] = stock_status
    result["understocked_flag"] = understocked
    result["overstocked_flag"] = overstocked
    result["inventory_action"] = inventory_action
    result["price_adjustment_pct"] = price_adjustment_pct
    result["reorder_point"] = reorder_point
    result["safety_stock"] = safety_stock
    result["recommended_order_qty"] = recommended_order
    result["stock_value"] = np.round(stock_value, 2)
    # Ensure inventory_action and price_adjustment_pct are written as correct types
    result["inventory_action"] = [str(a).lower() for a in inventory_action]
    result["price_adjustment_pct"] = price_adjustment_pct
    result["reorder_point"] = np.round(reorder_point, 1)
    result["safety_stock"] = np.round(safety_stock, 1)
    result["recommended_order_qty"] = np.round(recommended_order, 0).astype(int)

    # â”€â”€â”€ Build report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    report.understocked_count = int(np.sum(understocked))
    report.overstocked_count = int(np.sum(overstocked))
    report.healthy_count = n - report.understocked_count - report.overstocked_count

    # Actions
    action_arr = np.array(inventory_action, dtype=str)
    report.actions_increase = int(np.sum(action_arr == "increase"))
    report.actions_discount = int(np.sum(action_arr == "discount"))
    report.actions_maintain = int(np.sum(action_arr == "maintain"))
    report.actions_monitor = int(np.sum(action_arr == "monitor"))

    # Averages
    report.avg_days_of_cover = float(np.nanmean(days_of_cover))
    report.avg_inventory_score = float(np.nanmean(inventory_score))

    # Risk distribution
    risk_series = pd.Series(risk_level)
    report.risk_distribution = risk_series.value_counts().to_dict()

    # Stock status distribution
    status_series = pd.Series(stock_status)
    report.stock_status_distribution = status_series.value_counts().to_dict()

    # Stock value
    report.total_stock_value = float(np.nansum(stock_value))

    # Products needing reorder
    report.products_needing_reorder = int(np.sum(inventory_level < reorder_point))

    logger.info(
        f"Inventory analysis complete: "
        f"{report.healthy_count} healthy, "
        f"{report.understocked_count} understocked, "
        f"{report.overstocked_count} overstocked, "
        f"actions: {report.actions_increase}â†‘ {report.actions_discount}â†“ {report.actions_maintain}â†’"
    )

    return result, report


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# BATCH PROCESSING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def batch_analyze_inventory(
    df: pd.DataFrame,
    chunk_size: int = 5000,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, InventoryReport]:
    """
    Run inventory analysis in chunks for very large datasets.

    Args:
        df: Product DataFrame.
        chunk_size: Products per chunk.
        **kwargs: Additional args for analyze_inventory().

    Returns:
        Tuple of (enriched DataFrame, merged InventoryReport).
    """
    total = len(df)
    logger.info(f"Batch inventory analysis: {total} products, chunk={chunk_size}")

    if total <= chunk_size:
        return analyze_inventory(df, **kwargs)

    chunks = [df.iloc[i:i + chunk_size].copy() for i in range(0, total, chunk_size)]
    chunk_results: List[pd.DataFrame] = []
    combined_report = InventoryReport()
    combined_report.total_products = total

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_result, chunk_report = analyze_inventory(chunk, inplace=True, **kwargs)
        chunk_results.append(chunk_result)

        combined_report.understocked_count += chunk_report.understocked_count
        combined_report.overstocked_count += chunk_report.overstocked_count
        combined_report.healthy_count += chunk_report.healthy_count
        combined_report.actions_increase += chunk_report.actions_increase
        combined_report.actions_discount += chunk_report.actions_discount
        combined_report.actions_maintain += chunk_report.actions_maintain
        combined_report.actions_monitor += chunk_report.actions_monitor
        combined_report.total_stock_value += chunk_report.total_stock_value
        combined_report.products_needing_reorder += chunk_report.products_needing_reorder

    result_df = pd.concat(chunk_results, ignore_index=True)
    combined_report.avg_days_of_cover = float(result_df["days_of_cover"].mean())
    combined_report.avg_inventory_score = float(result_df["inventory_score"].mean())
    combined_report.risk_distribution = result_df["inventory_risk"].value_counts().to_dict()
    combined_report.stock_status_distribution = result_df["stock_status"].value_counts().to_dict()

    return result_df, combined_report


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CLASS-BASED API
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class InventoryEngine:
    """
    Inventory pricing optimisation engine.

    Analyses every product's inventory position alongside demand to generate
    pricing recommendations based on a 2Ã—2 decision matrix:

        Low stock + High demand  â†’ â†‘ Increase price (scarcity premium)
        Low stock + Low demand   â†’ â˜… Monitor (natural run-down)
        High stock + High demand â†’ â†’ Maintain (demand will clear it)
        High stock + Low demand  â†’ â†“ Discount (clearance)

    Also computes inventory health score (0-100), risk level, stock status,
    reorder points and safety stock.

    All logic is vectorised for 1000+ products.

    Usage:
        engine = InventoryEngine()
        df, report = engine.analyze(dataframe)
        print(report.summary())
        print(df["inventory_action"].iloc[0])
        print(df["price_adjustment_pct"].iloc[0])
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the inventory engine.

        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()
        self._last_report: Optional[InventoryReport] = None

    @property
    def last_report(self) -> Optional[InventoryReport]:
        """Get the report from the most recent analysis."""
        return self._last_report

    def analyze(
        self,
        df: pd.DataFrame,
        inventory_col: str = "inventory_level",
        sales_col: str = "sales_volume",
        demand_col: str = "demand_trend",
        price_col: str = "current_price",
        cost_col: str = "cost_price",
        inplace: bool = False,
    ) -> Tuple[pd.DataFrame, InventoryReport]:
        """
        Run inventory + pricing optimisation analysis on all products.

        Args:
            df: Product DataFrame.
            inventory_col: Inventory level column.
            sales_col: Sales volume column.
            demand_col: Demand trend column (0-1).
            price_col: Current price column.
            cost_col: Cost price column.
            inplace: If True, modifies df in place.

        Returns:
            Tuple of (enriched DataFrame, InventoryReport).
        """
        result_df, report = analyze_inventory(
            df,
            inventory_col=inventory_col,
            sales_col=sales_col,
            demand_col=demand_col,
            price_col=price_col,
            cost_col=cost_col,
            inplace=inplace,
        )
        self._last_report = report
        return result_df, report

    def analyze_single(
        self,
        inventory_level: float,
        sales_volume: float,
        demand_trend: float = 0.5,
        current_price: float = 100.0,
        cost_price: float = 60.0,
        product_name: str = "",
    ) -> InventoryAnalysis:
        """
        Analyse inventory + pricing for a single product.

        Args:
            inventory_level: Current stock units.
            sales_volume: Monthly sales units.
            demand_trend: Demand signal (0-1).
            current_price: Current selling price.
            cost_price: Unit cost.
            product_name: Product name.

        Returns:
            InventoryAnalysis with results.
        """
        row = {
            "inventory_level": inventory_level,
            "sales_volume": sales_volume,
            "demand_trend": demand_trend,
            "current_price": current_price,
            "cost_price": cost_price,
            "product_name": product_name,
        }
        df = pd.DataFrame([row])
        result_df, _ = self.analyze(df, inplace=False)

        if result_df.empty:
            return InventoryAnalysis(product_name=product_name)

        row_out = result_df.iloc[0]

        return InventoryAnalysis(
            product_id=str(row_out.get("product_id", "")),
            product_name=str(row_out.get("product_name", "")),
            inventory_level=float(row_out.get("inventory_level", 0)),
            sales_volume=float(row_out.get("sales_volume", 0)),
            days_of_cover=float(row_out.get("days_of_cover", 0)),
            inventory_score=float(row_out.get("inventory_score", 50)),
            inventory_risk=str(row_out.get("inventory_risk", "MEDIUM")),
            stock_status=str(row_out.get("stock_status", "Healthy")),
            inventory_action=str(row_out.get("inventory_action", "maintain")),
            price_adjustment_pct=float(row_out.get("price_adjustment_pct", 0.0)),
            demand_trend=float(row_out.get("demand_trend", 0.5)),
            reorder_point=float(row_out.get("reorder_point", 0)),
            safety_stock=float(row_out.get("safety_stock", 0)),
            recommended_order_qty=float(row_out.get("recommended_order_qty", 0)),
            is_understocked=bool(row_out.get("understocked_flag", False)),
            is_overstocked=bool(row_out.get("overstocked_flag", False)),
        )

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last analysis."""
        if self._last_report is None:
            return "No analysis performed yet."
        return self._last_report.summary()
