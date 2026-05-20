"""
Alert System Module

Configurable threshold-based alerting engine that monitors pricing,
inventory, and sales metrics. Generates actionable alerts for
business users.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.config import AppConfig
from utils.helpers import format_currency, safe_divide

logger = logging.getLogger("ai_pricing.alerts")


class AlertEngine:
    """
    Threshold-based alerting engine for pricing intelligence.

    Monitors:
    - Price changes (increase/decrease beyond thresholds)
    - Margin erosion
    - Competitive position shifts
    - Inventory anomalies (low stock, overstocked)
    - Sales declines
    - Risk threshold breaches

    Generates categorized alerts with severity levels.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the alert engine.

        Args:
            config: Application configuration with alert thresholds
        """
        self.config = config or AppConfig()
        self.alerts: List[Dict[str, Any]] = []

    def generate_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate all alerts based on current data and thresholds.

        Args:
            df: Full analysis DataFrame

        Returns:
            List of alert dictionaries with category, severity, message
        """
        self.alerts = []

        self._check_margin_alerts(df)
        self._check_competitor_alerts(df)
        self._check_inventory_alerts(df)
        self._check_sales_alerts(df)
        self._check_risk_alerts(df)

        # Sort by severity
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        self.alerts.sort(
            key=lambda x: severity_order.get(x.get("severity", "Low"), 99)
        )

        logger.info(
            f"Generated {len(self.alerts)} alerts "
            f"({sum(1 for a in self.alerts if a['severity'] in ('Critical', 'High'))} critical/high)"
        )
        return self.alerts

    def _check_margin_alerts(self, df: pd.DataFrame) -> None:
        """
        Check for margin-related alerts.

        Args:
            df: DataFrame with profit_margin column
        """
        if "profit_margin" not in df.columns:
            logger.debug("profit_margin column not found - skipping margin alerts")
            return

        try:
            # Low margin alerts
            low_margin = df[
                df["profit_margin"] < self.config.min_margin
            ]
            for _, row in low_margin.iterrows():
                margin_pct = row["profit_margin"] * 100
                self.alerts.append(
                    {
                        "category": "Margin",
                        "severity": "High" if margin_pct < 5 else "Medium",
                        "product_id": row.get("product_id", ""),
                        "product_name": row.get("product_name", "Unknown"),
                        "message": (
                            f"Critical margin: {margin_pct:.1f}% on "
                            f"{row.get('product_name', 'Unknown')} "
                            f"(Price: ${row.get('current_price', 0):.2f}, "
                            f"Cost: ${row.get('cost_price', 0):.2f})"
                        ),
                        "value": round(margin_pct, 1),
                        "threshold": self.config.min_margin * 100,
                    }
                )
            logger.debug(f"Generated {len(low_margin)} margin alerts")
        except Exception as e:
            logger.error(f"Error checking margin alerts: {e}", exc_info=True)

    def _check_competitor_alerts(self, df: pd.DataFrame) -> None:
        """
        Check for competitor pricing alerts.

        Args:
            df: DataFrame with competitive analysis columns
        """
        if "price_vs_competitor" not in df.columns:
            logger.debug("price_vs_competitor column not found - skipping competitor alerts")
            return

        try:
            # Priced significantly above competitor
            above_threshold = df[
                df["price_vs_competitor"]
                > (1 + self.config.price_above_competitor_threshold)
            ]
            for _, row in above_threshold.iterrows():
                self.alerts.append(
                    {
                        "category": "Competitor",
                        "severity": "Medium",
                        "product_id": row.get("product_id", ""),
                        "product_name": row.get("product_name", "Unknown"),
                        "message": (
                            f"Priced {((row['price_vs_competitor'] - 1) * 100):.0f}% above "
                            f"competitor (${row.get('competitor_price', 0):.2f}) - "
                            f"{row.get('product_name', 'Unknown')}"
                        ),
                        "value": round(row["price_vs_competitor"], 2),
                        "threshold": 1 + self.config.price_above_competitor_threshold,
                    }
                )
            logger.debug(f"Generated {len(above_threshold)} competitor alerts")
        except Exception as e:
            logger.error(f"Error checking competitor alerts: {e}", exc_info=True)

    def _check_inventory_alerts(self, df: pd.DataFrame) -> None:
        """
        Check for inventory-related alerts.

        Args:
            df: DataFrame with inventory analysis columns
        """
        if "days_of_cover" not in df.columns:
            logger.debug("days_of_cover column not found - skipping inventory alerts")
            return

        try:
            # Low stock alerts
            low_stock = df[df["days_of_cover"] < self.config.inventory_low_threshold]
            for _, row in low_stock.iterrows():
                self.alerts.append(
                    {
                        "category": "Inventory",
                        "severity": "High" if row["days_of_cover"] < 7 else "Medium",
                        "product_id": row.get("product_id", ""),
                        "product_name": row.get("product_name", "Unknown"),
                        "message": (
                            f"Low stock: {row.get('inventory_level', 0):.0f} units "
                            f"({row['days_of_cover']:.0f} days cover) - "
                            f"{row.get('product_name', 'Unknown')}"
                        ),
                        "value": round(row["days_of_cover"], 0),
                        "threshold": self.config.inventory_low_threshold,
                    }
                )

            # Overstocked inventory alerts
            overstocked = df[
                df["days_of_cover"] > self.config.inventory_excess_threshold
            ]
            for _, row in overstocked.iterrows():
                self.alerts.append(
                    {
                        "category": "Inventory",
                        "severity": "Medium",
                        "product_id": row.get("product_id", ""),
                        "product_name": row.get("product_name", "Unknown"),
                        "message": (
                            f"Overstocked inventory: {row.get('inventory_level', 0):.0f} units "
                            f"({row['days_of_cover']:.0f} days cover) - "
                            f"{row.get('product_name', 'Unknown')}"
                        ),
                        "value": round(row["days_of_cover"], 0),
                        "threshold": self.config.inventory_excess_threshold,
                    }
                )
            logger.debug(f"Generated {len(low_stock)} low-stock and {len(overstocked)} overstocked alerts")
        except Exception as e:
            logger.error(f"Error checking inventory alerts: {e}", exc_info=True)

    def _check_sales_alerts(self, df: pd.DataFrame) -> None:
        """
        Check for sales-related alerts.

        Args:
            df: DataFrame with sales data
        """
        if "sales_volume" not in df.columns:
            logger.debug("sales_volume column not found - skipping sales alerts")
            return

        try:
            # Low sales alerts (bottom 10% by volume)
            if len(df) >= 10:
                threshold = df["sales_volume"].quantile(0.1)
                low_sales = df[df["sales_volume"] <= threshold]

                for _, row in low_sales.iterrows():
                    self.alerts.append(
                        {
                            "category": "Sales",
                            "severity": "Medium",
                            "product_id": row.get("product_id", ""),
                            "product_name": row.get("product_name", "Unknown"),
                            "message": (
                                f"Low sales volume: {row['sales_volume']:.0f} units "
                                f"(bottom 10%) - {row.get('product_name', 'Unknown')}"
                            ),
                            "value": round(row["sales_volume"], 0),
                            "threshold": round(threshold, 0),
                        }
                    )
                logger.debug(f"Generated {len(low_sales)} sales alerts")
        except Exception as e:
            logger.error(f"Error checking sales alerts: {e}", exc_info=True)

    def _check_risk_alerts(self, df: pd.DataFrame) -> None:
        """
        Check for risk threshold breaches.

        Args:
            df: DataFrame with risk scores
        """
        if "composite_risk_score" not in df.columns:
            logger.debug("composite_risk_score column not found - skipping risk alerts")
            return

        try:
            high_risk = df[df["composite_risk_score"] >= self.config.high_risk_score]
            for _, row in high_risk.iterrows():
                self.alerts.append(
                    {
                    "category": "Risk",
                    "severity": "Critical" if row["composite_risk_score"] >= 80 else "High",
                    "product_id": row.get("product_id", ""),
                    "product_name": row.get("product_name", "Unknown"),
                    "message": (
                        f"High risk score: {row['composite_risk_score']:.0f}/100 "
                        f"(Primary: {row.get('primary_risk_factor', 'Unknown')}) - "
                        f"{row.get('product_name', 'Unknown')}"
                    ),
                    "value": round(row["composite_risk_score"], 1),
                    "threshold": self.config.high_risk_score,
                }
            )
            logger.debug(f"Generated {len(high_risk)} risk alerts")
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}", exc_info=True)

    def get_alerts_by_severity(
        self, severity: str
    ) -> List[Dict[str, Any]]:
        """
        Filter alerts by severity level.

        Args:
            severity: Severity level to filter ('Critical', 'High', 'Medium', 'Low')

        Returns:
            Filtered list of alerts
        """
        return [a for a in self.alerts if a.get("severity") == severity]

    def get_alerts_by_category(
        self, category: str
    ) -> List[Dict[str, Any]]:
        """
        Filter alerts by category.

        Args:
            category: Alert category ('Margin', 'Competitor', 'Inventory', 'Sales', 'Risk')

        Returns:
            Filtered list of alerts
        """
        return [
            a for a in self.alerts if a.get("category") == category
        ]

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all generated alerts.

        Returns:
            Dictionary with alert counts by severity and category
        """
        summary: Dict[str, Any] = {
            "total_alerts": len(self.alerts),
            "by_severity": {},
            "by_category": {},
            "critical_alerts": [],
        }

        for alert in self.alerts:
            severity = alert.get("severity", "Unknown")
            category = alert.get("category", "Unknown")

            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

            if severity == "Critical":
                summary["critical_alerts"].append(alert)

        return summary