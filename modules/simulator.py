"""
Pricing What-If Simulator Module

Allows users to modify price, demand, and inventory parameters to simulate
the impact on profit, revenue, and margin for single products or bulk batches.

Features:
  - Single product simulation with interactive parameter sliders
  - Bulk simulation for multiple products / scenarios
  - Scenario comparison with side-by-side tables
  - Sensitivity analysis (what drives profit the most?)
  - Streamlit-ready output (charts, tables, formatted metrics)

Core Functions:
  - simulate_single()        — One product, one or more scenarios
  - simulate_bulk()          — Multiple products × multiple scenarios
  - compare_scenarios()      — Side-by-side comparison of scenario results
  - sensitivity_analysis()   — Tornado chart data for key drivers

All logic is vectorised for 1000+ products.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.helpers import safe_divide, round_half_up

logger = logging.getLogger("ai_pricing.simulator")


# ─── Constants ─────────────────────────────────────────────────────────────

# Default scenario templates
SCENARIO_TEMPLATES: Dict[str, Dict[str, float]] = {
    "base": {
        "price_change_pct": 0.0,
        "demand_change_pct": 0.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 0.0,
    },
    "price_increase_10": {
        "price_change_pct": 10.0,
        "demand_change_pct": 0.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 0.0,
    },
    "price_decrease_10": {
        "price_change_pct": -10.0,
        "demand_change_pct": 0.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 0.0,
    },
    "demand_surge_20": {
        "price_change_pct": 0.0,
        "demand_change_pct": 20.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 0.0,
    },
    "demand_drop_20": {
        "price_change_pct": 0.0,
        "demand_change_pct": -20.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 0.0,
    },
    "inventory_double": {
        "price_change_pct": 0.0,
        "demand_change_pct": 0.0,
        "inventory_change_pct": 100.0,
        "cost_change_pct": 0.0,
    },
    "cost_increase_15": {
        "price_change_pct": 0.0,
        "demand_change_pct": 0.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 15.0,
    },
    "optimistic": {
        "price_change_pct": 15.0,
        "demand_change_pct": 25.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": -10.0,
    },
    "pessimistic": {
        "price_change_pct": -15.0,
        "demand_change_pct": -25.0,
        "inventory_change_pct": 0.0,
        "cost_change_pct": 15.0,
    },
}


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class SimulatedMetric:
    """
    A single metric result from a simulation scenario.
    """
    label: str = ""           # e.g. "Revenue"
    base_value: float = 0.0
    simulated_value: float = 0.0
    change_value: float = 0.0
    change_pct: float = 0.0
    currency: bool = True     # Whether to format as currency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.label,
            "base": round(self.base_value, 2),
            "simulated": round(self.simulated_value, 2),
            "change": round(self.change_value, 2),
            "change_pct": round(self.change_pct, 2),
        }


@dataclass
class ScenarioResult:
    """
    Result of simulating one scenario for one or more products.
    """
    scenario_name: str = ""
    scenario_params: Dict[str, float] = field(default_factory=dict)

    # Aggregated for batch: summed across products
    total_revenue: float = 0.0
    total_profit: float = 0.0
    avg_margin: float = 0.0
    total_volume: float = 0.0
    total_stock_value: float = 0.0
    days_of_cover: float = 0.0

    # Change vs base
    revenue_change_pct: float = 0.0
    profit_change_pct: float = 0.0
    margin_change: float = 0.0  # absolute % change

    # Per-product details (for single product or summary)
    product_count: int = 0
    metrics: List[SimulatedMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "params": self.scenario_params,
            "total_revenue": round(self.total_revenue, 2),
            "total_profit": round(self.total_profit, 2),
            "avg_margin": round(self.avg_margin, 2),
            "total_volume": round(self.total_volume, 0),
            "revenue_change_pct": round(self.revenue_change_pct, 2),
            "profit_change_pct": round(self.profit_change_pct, 2),
            "margin_change": round(self.margin_change, 2),
            "product_count": self.product_count,
        }

    def summary_line(self) -> str:
        """One-line summary of the scenario result."""
        rev_str = f"{self.revenue_change_pct:+.1f}%" if self.revenue_change_pct != 0 else "no change"
        profit_str = f"{self.profit_change_pct:+.1f}%" if self.profit_change_pct != 0 else "no change"
        return (
            f"[{self.scenario_name}] "
            f"Rev: ${self.total_revenue:,.0f} ({rev_str}), "
            f"Profit: ${self.total_profit:,.0f} ({profit_str}), "
            f"Margin: {self.avg_margin:.1f}%"
        )


@dataclass
class SimulationReport:
    """
    Aggregate report for a batch what-if simulation.
    """
    total_products: int = 0
    total_scenarios: int = 0
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    best_scenario: str = ""
    worst_scenario: str = ""
    best_profit_change: float = 0.0
    worst_profit_change: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "products": self.total_products,
            "scenarios": self.total_scenarios,
            "best": self.best_scenario,
            "worst": self.worst_scenario,
            "best_profit_change_pct": round(self.best_profit_change, 1),
            "worst_profit_change_pct": round(self.worst_profit_change, 1),
            "results": [r.to_dict() for r in self.scenario_results],
        }

    def summary(self) -> str:
        lines = [
            f"  Products:       {self.total_products}",
            f"  Scenarios:      {self.total_scenarios}",
            f"  Best scenario:  {self.best_scenario} ({self.best_profit_change:+.1f}% profit)",
            f"  Worst scenario: {self.worst_scenario} ({self.worst_profit_change:+.1f}% profit)",
        ]
        for r in self.scenario_results:
            lines.append(f"  {r.summary_line()}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CORE SIMULATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _apply_price_elasticity(
    price_change_pct: float,
    price_elasticity: float,
    cap: float = 0.40,
) -> float:
    """
    Estimate volume change from a price change using elasticity.

    volume_change_pct = elasticity × price_change_pct
    Capped to avoid unrealistic projections.

    Args:
        price_change_pct: % change in price (e.g. 10.0 for +10%).
        price_elasticity: Price elasticity coefficient (e.g. -1.5).
        cap: Maximum absolute volume change fraction (default 0.40).

    Returns:
        Estimated % change in volume.
    """
    raw = price_elasticity * (price_change_pct / 100.0)
    return np.clip(raw, -cap, cap) * 100


def _simulate_product(
    current_price: float,
    cost_price: float,
    sales_volume: float,
    inventory_level: float,
    demand_trend: float,
    price_elasticity: float,
    scenario: Dict[str, float],
) -> Dict[str, float]:
    """
    Simulate the impact of a scenario on a single product.

    Args:
        current_price: Current selling price.
        cost_price: Unit cost.
        sales_volume: Monthly sales units.
        inventory_level: Current stock units.
        demand_trend: Demand signal (0-1).
        price_elasticity: Price elasticity coefficient.
        scenario: Dict with % changes:
            price_change_pct, demand_change_pct,
            inventory_change_pct, cost_change_pct.

    Returns:
        Dict of simulated metrics.
    """
    p_pct = scenario.get("price_change_pct", 0.0)
    d_pct = scenario.get("demand_change_pct", 0.0)
    i_pct = scenario.get("inventory_change_pct", 0.0)
    c_pct = scenario.get("cost_change_pct", 0.0)

    # New price
    new_price = current_price * (1.0 + p_pct / 100.0)

    # New cost
    new_cost = cost_price * (1.0 + c_pct / 100.0)

    # Volume change: demand change + elasticity from price change
    vol_from_elasticity = _apply_price_elasticity(p_pct, price_elasticity)
    total_vol_change_pct = d_pct + vol_from_elasticity
    new_volume = sales_volume * (1.0 + total_vol_change_pct / 100.0)
    new_volume = max(0, new_volume)

    # New inventory
    new_inventory = inventory_level * (1.0 + i_pct / 100.0)
    new_inventory = max(0, new_inventory)

    # Revenue
    base_revenue = current_price * sales_volume
    new_revenue = new_price * new_volume

    # Profit
    base_profit = (current_price - cost_price) * sales_volume
    new_profit = (new_price - new_cost) * new_volume

    # Margin
    base_margin = (
        (current_price - cost_price) / current_price * 100
        if current_price > 0 else 0.0
    )
    new_margin = (
        (new_price - new_cost) / new_price * 100
        if new_price > 0 else 0.0
    )

    # Days of cover
    daily_sales = new_volume / 30.0 if new_volume > 0 else 0.001
    new_doc = new_inventory / daily_sales if daily_sales > 0 else 999.0

    # Stock value
    new_stock_value = new_inventory * new_cost

    return {
        "new_price": new_price,
        "new_cost": new_cost,
        "new_volume": new_volume,
        "new_inventory": new_inventory,
        "new_revenue": new_revenue,
        "new_profit": new_profit,
        "new_margin": new_margin,
        "new_days_of_cover": new_doc,
        "new_stock_value": new_stock_value,
        "base_revenue": base_revenue,
        "base_profit": base_profit,
        "base_margin": base_margin,
        "volume_change_pct": total_vol_change_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE PRODUCT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_single(
    current_price: float = 100.0,
    cost_price: float = 60.0,
    sales_volume: float = 1000.0,
    inventory_level: float = 500.0,
    demand_trend: float = 0.5,
    price_elasticity: float = -1.5,
    scenarios: Optional[List[Dict[str, Any]]] = None,
    product_name: str = "Product",
) -> Tuple[List[ScenarioResult], SimulationReport]:
    """
    Simulate one or more what-if scenarios for a single product.

    Each scenario is a dict with:
    - name: Scenario label (optional, auto-generated if missing)
    - price_change_pct: % change in price
    - demand_change_pct: % change in base demand
    - inventory_change_pct: % change in inventory level
    - cost_change_pct: % change in unit cost

    Args:
        current_price: Current selling price.
        cost_price: Unit cost.
        sales_volume: Monthly sales volume.
        inventory_level: Current stock level.
        demand_trend: Demand signal (0-1).
        price_elasticity: Price elasticity coefficient.
        scenarios: List of scenario dicts. If None, uses SCENARIO_TEMPLATES.
        product_name: Product name for display.

    Returns:
        Tuple of (list of ScenarioResult, SimulationReport).
    """
    if scenarios is None:
        scenarios = [
            {"name": name, **params}
            for name, params in SCENARIO_TEMPLATES.items()
        ]

    # Always include a base scenario for comparison if not present
    has_base = any(s.get("name") == "base" for s in scenarios)
    if not has_base:
        scenarios = [{"name": "base", **SCENARIO_TEMPLATES["base"]}] + scenarios

    report = SimulationReport()
    report.total_products = 1
    report.total_scenarios = len(scenarios)

    results: List[ScenarioResult] = []

    for scenario in scenarios:
        sname = scenario.get("name", "Scenario")
        sparams = {
            "price_change_pct": scenario.get("price_change_pct", 0.0),
            "demand_change_pct": scenario.get("demand_change_pct", 0.0),
            "inventory_change_pct": scenario.get("inventory_change_pct", 0.0),
            "cost_change_pct": scenario.get("cost_change_pct", 0.0),
        }

        sim = _simulate_product(
            current_price, cost_price, sales_volume,
            inventory_level, demand_trend, price_elasticity,
            sparams,
        )

        # Build ScenarioResult
        result = ScenarioResult(
            scenario_name=sname,
            scenario_params=sparams,
            total_revenue=sim["new_revenue"],
            total_profit=sim["new_profit"],
            avg_margin=sim["new_margin"],
            total_volume=sim["new_volume"],
            total_stock_value=sim["new_stock_value"],
            days_of_cover=sim["new_days_of_cover"],
            revenue_change_pct=safe_divide(
                sim["new_revenue"] - sim["base_revenue"], sim["base_revenue"]
            ) * 100,
            profit_change_pct=safe_divide(
                sim["new_profit"] - sim["base_profit"], abs(sim["base_profit"])
            ) * 100 if sim["base_profit"] != 0 else 0.0,
            margin_change=sim["new_margin"] - sim["base_margin"],
            product_count=1,
        )

        # Build metric list for display
        metrics = [
            SimulatedMetric("Price", current_price, sim["new_price"], sim["new_price"] - current_price,
                           (sim["new_price"] - current_price) / current_price * 100, currency=True),
            SimulatedMetric("Cost", cost_price, sim["new_cost"], sim["new_cost"] - cost_price,
                           (sim["new_cost"] - cost_price) / cost_price * 100, currency=True),
            SimulatedMetric("Volume", sales_volume, sim["new_volume"], sim["new_volume"] - sales_volume,
                           sim["volume_change_pct"], currency=False),
            SimulatedMetric("Revenue", sim["base_revenue"], sim["new_revenue"],
                           sim["new_revenue"] - sim["base_revenue"],
                           result.revenue_change_pct, currency=True),
            SimulatedMetric("Profit", sim["base_profit"], sim["new_profit"],
                           sim["new_profit"] - sim["base_profit"],
                           result.profit_change_pct, currency=True),
            SimulatedMetric("Margin %", sim["base_margin"], sim["new_margin"],
                           result.margin_change, 0.0, currency=False),
            SimulatedMetric("Inventory", inventory_level, sim["new_inventory"],
                           sim["new_inventory"] - inventory_level,
                           sparams.get("inventory_change_pct", 0.0), currency=False),
            SimulatedMetric("Days of Cover", 
                           safe_divide(inventory_level, max(sales_volume / 30.0, 0.01)),
                           sim["new_days_of_cover"],
                           sim["new_days_of_cover"] - safe_divide(inventory_level, max(sales_volume / 30.0, 0.01)),
                           0.0, currency=False),
        ]
        result.metrics = metrics
        results.append(result)

    report.scenario_results = results

    # Find best / worst by profit change
    profit_changes = [r.profit_change_pct if r.scenario_name != "base" else -9999 for r in results]
    if profit_changes:
        best_idx = int(np.argmax(profit_changes))
        worst_idx = int(np.argmin(profit_changes))
        report.best_scenario = results[best_idx].scenario_name
        report.worst_scenario = results[worst_idx].scenario_name
        report.best_profit_change = profit_changes[best_idx]
        report.worst_profit_change = profit_changes[worst_idx]

    logger.info(
        f"simulate_single: {product_name}, {len(scenarios)} scenarios, "
        f"best={report.best_scenario} ({report.best_profit_change:+.1f}%)"
    )

    return results, report


# ═══════════════════════════════════════════════════════════════════════════
# BULK SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def simulate_bulk(
    df: pd.DataFrame,
    scenarios: Optional[List[Dict[str, Any]]] = None,
    price_col: str = "current_price",
    cost_col: str = "cost_price",
    volume_col: str = "sales_volume",
    inventory_col: str = "inventory_level",
    demand_col: str = "demand_trend",
    elasticity_col: str = "price_elasticity",
    name_col: str = "product_name",
) -> Tuple[pd.DataFrame, SimulationReport]:
    """
    Run what-if simulations for MULTIPLE products across MULTIPLE scenarios.

    Returns a DataFrame with one row per (product × scenario) combination,
    plus a SimulationReport with aggregate results.

    Args:
        df: DataFrame with product data.
        scenarios: List of scenario dicts. If None, uses SCENARIO_TEMPLATES.
        price_col: Column with current price.
        cost_col: Column with cost price.
        volume_col: Column with sales volume.
        inventory_col: Column with inventory level.
        demand_col: Column with demand trend.
        elasticity_col: Column with price elasticity.
        name_col: Column with product name.

    Returns:
        Tuple of:
        - DataFrame with columns: product_id, product_name, scenario_name,
          current_price, new_price, base_revenue, new_revenue, base_profit,
          new_profit, base_margin, new_margin, volume_change_pct,
          revenue_change_pct, profit_change_pct, margin_change
        - SimulationReport with aggregate statistics.
    """
    if scenarios is None:
        scenarios = [
            {"name": name, **params}
            for name, params in SCENARIO_TEMPLATES.items()
        ]

    # Ensure base scenario
    has_base = any(s.get("name") == "base" for s in scenarios)
    if not has_base:
        scenarios = [{"name": "base", **SCENARIO_TEMPLATES["base"]}] + scenarios

    n = len(df)
    n_scenarios = len(scenarios)
    total_rows = n * n_scenarios

    logger.info(
        f"simulate_bulk: {n} products × {n_scenarios} scenarios = {total_rows} simulations"
    )

    report = SimulationReport()
    report.total_products = n
    report.total_scenarios = n_scenarios

    # Extract arrays
    prices = pd.to_numeric(df[price_col], errors="coerce").fillna(0).values if price_col in df.columns else np.full(n, 100.0)
    costs = pd.to_numeric(df[cost_col], errors="coerce").fillna(0).values if cost_col in df.columns else np.full(n, 60.0)
    volumes = pd.to_numeric(df[volume_col], errors="coerce").fillna(0).values if volume_col in df.columns else np.full(n, 1000.0)
    inventories = pd.to_numeric(df[inventory_col], errors="coerce").fillna(0).values if inventory_col in df.columns else np.full(n, 500.0)
    demands = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.5).values if demand_col in df.columns else np.full(n, 0.5)
    elasticities = pd.to_numeric(df[elasticity_col], errors="coerce").fillna(-1.5).values if elasticity_col in df.columns else np.full(n, -1.5)

    product_ids = df.index.values if "product_id" not in df.columns else df["product_id"].values
    product_names = df[name_col].values if name_col in df.columns else np.full(n, "")

    # Build result rows
    rows: List[Dict[str, Any]] = []
    scenario_results_list: List[ScenarioResult] = []

    for s_idx, scenario in enumerate(scenarios):
        sname = scenario.get("name", f"Scenario_{s_idx}")
        sparams = {
            "price_change_pct": scenario.get("price_change_pct", 0.0),
            "demand_change_pct": scenario.get("demand_change_pct", 0.0),
            "inventory_change_pct": scenario.get("inventory_change_pct", 0.0),
            "cost_change_pct": scenario.get("cost_change_pct", 0.0),
        }

        # Aggregate accumulators
        total_rev = 0.0
        total_profit = 0.0
        total_vol = 0.0
        margins: List[float] = []
        base_rev_total = 0.0
        base_profit_total = 0.0
        base_margins: List[float] = []

        for i in range(n):
            sim = _simulate_product(
                float(prices[i]), float(costs[i]), float(volumes[i]),
                float(inventories[i]), float(demands[i]), float(elasticities[i]),
                sparams,
            )
            pid = str(product_ids[i]) if i < len(product_ids) else str(i)
            pname = str(product_names[i]) if i < len(product_names) else ""

            rev_change = safe_divide(sim["new_revenue"] - sim["base_revenue"], sim["base_revenue"]) * 100
            profit_change = safe_divide(sim["new_profit"] - sim["base_profit"], abs(sim["base_profit"])) * 100 if sim["base_profit"] != 0 else 0.0
            margin_change = sim["new_margin"] - sim["base_margin"]

            rows.append({
                "product_id": pid,
                "product_name": pname,
                "scenario_name": sname,
                "current_price": round(float(prices[i]), 2),
                "new_price": round(sim["new_price"], 2),
                "base_revenue": round(sim["base_revenue"], 2),
                "new_revenue": round(sim["new_revenue"], 2),
                "base_profit": round(sim["base_profit"], 2),
                "new_profit": round(sim["new_profit"], 2),
                "base_margin": round(sim["base_margin"], 2),
                "new_margin": round(sim["new_margin"], 2),
                "volume_change_pct": round(sim["volume_change_pct"], 2),
                "revenue_change_pct": round(rev_change, 2),
                "profit_change_pct": round(profit_change, 2),
                "margin_change": round(margin_change, 2),
                "new_volume": round(sim["new_volume"], 0),
            })

            total_rev += sim["new_revenue"]
            total_profit += sim["new_profit"]
            total_vol += sim["new_volume"]
            margins.append(sim["new_margin"])
            base_rev_total += sim["base_revenue"]
            base_profit_total += sim["base_profit"]
            base_margins.append(sim["base_margin"])

        # Build scenario result
        avg_margin = float(np.mean(margins)) if margins else 0.0
        avg_base_margin = float(np.mean(base_margins)) if base_margins else 0.0

        sr = ScenarioResult(
            scenario_name=sname,
            scenario_params=sparams,
            total_revenue=total_rev,
            total_profit=total_profit,
            avg_margin=avg_margin,
            total_volume=total_vol,
            revenue_change_pct=safe_divide(total_rev - base_rev_total, base_rev_total) * 100,
            profit_change_pct=safe_divide(total_profit - base_profit_total, abs(base_profit_total)) * 100 if base_profit_total != 0 else 0.0,
            margin_change=avg_margin - avg_base_margin,
            product_count=n,
        )
        scenario_results_list.append(sr)

    report.scenario_results = scenario_results_list

    # Find best / worst
    non_base = [r for r in scenario_results_list if r.scenario_name != "base"]
    if non_base:
        profit_changes = [r.profit_change_pct for r in non_base]
        best_idx = int(np.argmax(profit_changes))
        worst_idx = int(np.argmin(profit_changes))
        report.best_scenario = non_base[best_idx].scenario_name
        report.worst_scenario = non_base[worst_idx].scenario_name
        report.best_profit_change = profit_changes[best_idx]
        report.worst_profit_change = profit_changes[worst_idx]

    result_df = pd.DataFrame(rows)

    logger.info(
        f"simulate_bulk complete: {len(result_df)} rows, "
        f"best={report.best_scenario} ({report.best_profit_change:+.1f}%), "
        f"worst={report.worst_scenario} ({report.worst_profit_change:+.1f}%)"
    )

    return result_df, report


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON & SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════

def compare_scenarios(
    results: List[ScenarioResult],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a side-by-side comparison table of scenario results.

    Args:
        results: List of ScenarioResult objects.
        metrics: List of metric keys to include.
                 Default: ["total_revenue", "total_profit", "avg_margin",
                          "revenue_change_pct", "profit_change_pct"].

    Returns:
        DataFrame with scenarios as rows and metrics as columns.
    """
    if metrics is None:
        metrics = [
            "total_revenue", "total_profit", "avg_margin",
            "revenue_change_pct", "profit_change_pct", "margin_change",
        ]

    comparison_data: List[Dict[str, Any]] = []
    for r in results:
        row = {"scenario": r.scenario_name}
        for m in metrics:
            row[m] = getattr(r, m, 0.0)
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Reorder: base first if present
    if "base" in df["scenario"].values:
        base_idx = df[df["scenario"] == "base"].index[0]
        base_row = df.loc[base_idx].to_dict()
        df = df.drop(base_idx)
        df = pd.concat([pd.DataFrame([base_row]), df], ignore_index=True)

    return df


def sensitivity_analysis(
    current_price: float = 100.0,
    cost_price: float = 60.0,
    sales_volume: float = 1000.0,
    inventory_level: float = 500.0,
    demand_trend: float = 0.5,
    price_elasticity: float = -1.5,
) -> pd.DataFrame:
    """
    Run a sensitivity analysis to determine which variable has the
    greatest impact on profit.

    Tests each variable (price, demand, inventory, cost) at ±10% and ±20%
    while holding others constant. Returns a DataFrame suitable for
    a tornado chart.

    Returns:
        DataFrame with columns: variable, change_pct, profit_change_pct,
        impact (absolute profit change).
    """
    base_sim = _simulate_product(
        current_price, cost_price, sales_volume,
        inventory_level, demand_trend, price_elasticity,
        SCENARIO_TEMPLATES["base"],
    )
    base_profit = base_sim["new_profit"]

    variables = [
        ("Price", "price_change_pct"),
        ("Demand", "demand_change_pct"),
        ("Inventory", "inventory_change_pct"),
        ("Cost", "cost_change_pct"),
    ]

    rows: List[Dict[str, Any]] = []

    for var_label, var_key in variables:
        for change in [-20.0, -10.0, 10.0, 20.0]:
            scenario = {"name": f"{var_label}_{change:+.0f}%", **SCENARIO_TEMPLATES["base"]}
            scenario[var_key] = change

            sim = _simulate_product(
                current_price, cost_price, sales_volume,
                inventory_level, demand_trend, price_elasticity,
                scenario,
            )

            profit_change = sim["new_profit"] - base_profit
            profit_change_pct = safe_divide(profit_change, abs(base_profit)) * 100 if base_profit != 0 else 0.0

            rows.append({
                "variable": var_label,
                "change_pct": change,
                "profit_change": round(profit_change, 2),
                "profit_change_pct": round(profit_change_pct, 2),
            })

    df = pd.DataFrame(rows)

    # Compute absolute impact for tornado sorting
    impact = df.groupby("variable")["profit_change_pct"].apply(
        lambda x: abs(x).max()
    ).sort_values(ascending=False)
    df["impact"] = df["variable"].map(impact)

    return df.sort_values(["impact", "variable", "change_pct"], ascending=[False, True, True])


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def format_scenario_table(results: List[ScenarioResult]) -> pd.DataFrame:
    """
    Format scenario results into a display-ready DataFrame for Streamlit.

    Colours: green for positive changes, red for negative.

    Args:
        results: List of ScenarioResult.

    Returns:
        Formatted DataFrame with styled columns.
    """
    comparison = compare_scenarios(results)
    
    # Format columns
    for col in ["total_revenue", "total_profit"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].apply(lambda x: f"${x:,.0f}")
    
    if "avg_margin" in comparison.columns:
        comparison["avg_margin"] = comparison["avg_margin"].apply(lambda x: f"{x:.1f}%")
    
    for col in ["revenue_change_pct", "profit_change_pct"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].apply(lambda x: f"{x:+.1f}%")
    
    if "margin_change" in comparison.columns:
        comparison["margin_change"] = comparison["margin_change"].apply(lambda x: f"{x:+.1f}pp")
    
    # Rename for display
    col_rename = {
        "scenario": "Scenario",
        "total_revenue": "Total Revenue",
        "total_profit": "Total Profit",
        "avg_margin": "Avg Margin",
        "revenue_change_pct": "Revenue Δ",
        "profit_change_pct": "Profit Δ",
        "margin_change": "Margin Δ",
        "total_volume": "Volume",
    }
    comparison = comparison.rename(columns={k: v for k, v in col_rename.items() if k in comparison.columns})
    
    return comparison


def format_single_product_results(
    results: List[ScenarioResult],
    product_name: str = "",
) -> pd.DataFrame:
    """
    Format single-product simulation into a detailed metric table.

    Each row is a metric (Price, Cost, Volume, Revenue, Profit, Margin),
    each column is a scenario.

    Args:
        results: List of ScenarioResult (from simulate_single).
        product_name: Product name for display.

    Returns:
        DataFrame with metrics as rows and scenarios as columns.
    """
    metric_names = ["Price", "Cost", "Volume", "Revenue", "Profit", "Margin %", "Inventory", "Days of Cover"]
    
    data: Dict[str, List[Any]] = {"Metric": metric_names}
    
    for r in results:
        col_name = r.scenario_name
        values: List[str] = []
        for m in metric_names:
            match = [mt for mt in r.metrics if mt.label == m]
            if match:
                mv = match[0]
                if mv.currency:
                    values.append(f"${mv.simulated_value:,.2f}")
                else:
                    values.append(f"{mv.simulated_value:,.1f}")
            else:
                values.append("-")
        data[col_name] = values
    
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API
# ═══════════════════════════════════════════════════════════════════════════

class PricingSimulator:
    """
    Interactive pricing what-if simulator.

    Allows users to modify price, demand, and inventory parameters
    to visualise the impact on profit, revenue, and margin.

    Supports single-product detailed analysis and bulk batch simulation
    for 1000+ products. Generates comparison tables and sensitivity
    analysis data ready for Streamlit dashboards.

    Usage (Streamlit integration):
        simulator = PricingSimulator()

        # Single product with interactive sliders
        scenarios = [
            {"name": "+10% Price", "price_change_pct": 10},
            {"name": "-10% Price", "price_change_pct": -10},
            {"name": "Optimistic", "price_change_pct": 15, "demand_change_pct": 20},
        ]
        results, report = simulator.simulate_single(
            current_price=79.99, cost_price=35.00,
            sales_volume=1240, inventory_level=850,
            demand_trend=0.85, price_elasticity=-1.8,
            scenarios=scenarios,
        )

        # Display in Streamlit
        st.dataframe(simulator.format_results(results))
        st.write(report.summary())

        # Bulk simulation for all products
        df_result, report = simulator.simulate_bulk(products_df)

        # Sensitivity analysis
        sens_df = simulator.sensitivity_analysis(current_price=79.99, cost_price=35.00)
        st.bar_chart(sens_df.pivot(index="variable", columns="change_pct", values="profit_change_pct"))
    """

    def __init__(self) -> None:
        """Initialize the pricing simulator."""
        self._last_report: Optional[SimulationReport] = None

    @property
    def last_report(self) -> Optional[SimulationReport]:
        """Get the report from the most recent simulation."""
        return self._last_report

    def simulate_single(
        self,
        current_price: float = 100.0,
        cost_price: float = 60.0,
        sales_volume: float = 1000.0,
        inventory_level: float = 500.0,
        demand_trend: float = 0.5,
        price_elasticity: float = -1.5,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        product_name: str = "Product",
    ) -> Tuple[List[ScenarioResult], SimulationReport]:
        """
        Simulate what-if scenarios for a single product.

        Args:
            current_price: Current selling price.
            cost_price: Unit cost.
            sales_volume: Monthly sales volume.
            inventory_level: Current stock level.
            demand_trend: Demand signal (0-1).
            price_elasticity: Price elasticity coefficient.
            scenarios: List of scenario dicts. Uses defaults if None.
            product_name: Product name for display.

        Returns:
            Tuple of (list of ScenarioResult, SimulationReport).
        """
        results, report = simulate_single(
            current_price=current_price,
            cost_price=cost_price,
            sales_volume=sales_volume,
            inventory_level=inventory_level,
            demand_trend=demand_trend,
            price_elasticity=price_elasticity,
            scenarios=scenarios,
            product_name=product_name,
        )
        self._last_report = report
        return results, report

    def simulate_bulk(
        self,
        df: pd.DataFrame,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        price_col: str = "current_price",
        cost_col: str = "cost_price",
        volume_col: str = "sales_volume",
        inventory_col: str = "inventory_level",
        demand_col: str = "demand_trend",
        elasticity_col: str = "price_elasticity",
        name_col: str = "product_name",
    ) -> Tuple[pd.DataFrame, SimulationReport]:
        """
        Run what-if simulations for multiple products across multiple scenarios.

        Args:
            df: Product DataFrame.
            scenarios: List of scenario dicts.
            price_col: Price column.
            cost_col: Cost column.
            volume_col: Volume column.
            inventory_col: Inventory column.
            demand_col: Demand column.
            elasticity_col: Elasticity column.
            name_col: Product name column.

        Returns:
            Tuple of (results DataFrame, SimulationReport).
        """
        result_df, report = simulate_bulk(
            df=df,
            scenarios=scenarios,
            price_col=price_col,
            cost_col=cost_col,
            volume_col=volume_col,
            inventory_col=inventory_col,
            demand_col=demand_col,
            elasticity_col=elasticity_col,
            name_col=name_col,
        )
        self._last_report = report
        return result_df, report

    def sensitivity_analysis(
        self,
        current_price: float = 100.0,
        cost_price: float = 60.0,
        sales_volume: float = 1000.0,
        inventory_level: float = 500.0,
        demand_trend: float = 0.5,
        price_elasticity: float = -1.5,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis to identify key profit drivers.

        Tests price, demand, inventory, and cost at ±10% and ±20%.

        Returns:
            DataFrame suitable for tornado chart.
        """
        return sensitivity_analysis(
            current_price=current_price,
            cost_price=cost_price,
            sales_volume=sales_volume,
            inventory_level=inventory_level,
            demand_trend=demand_trend,
            price_elasticity=price_elasticity,
        )

    def format_results(
        self,
        results: List[ScenarioResult],
        single_product: bool = True,
    ) -> pd.DataFrame:
        """
        Format simulation results into a display-ready DataFrame.

        Args:
            results: List of ScenarioResult.
            single_product: If True, shows per-metric breakdown.
                            If False, shows aggregate comparison.

        Returns:
            Formatted DataFrame for Streamlit display.
        """
        if single_product and results and any(r.metrics for r in results):
            return format_single_product_results(results)
        return format_scenario_table(results)

    def get_report_summary(self) -> str:
        """Get a human-readable summary of the last simulation."""
        if self._last_report is None:
            return "No simulation performed yet."
        return self._last_report.summary()