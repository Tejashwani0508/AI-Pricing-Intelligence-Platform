"""
Dashboard Visualizations Module

Comprehensive, production-grade Plotly chart factory for the Streamlit dashboard.
Provides modular plotting functions for KPIs, pricing, profit, risk, competition,
inventory, and demand analysis â€” all designed for 1000+ products.

Chart Types:
  - KPI Cards (indicator gauges)
  - Bar Charts (horizontal & vertical)
  - Pie / Donut Charts
  - Scatter Plots (with color/size encoding)
  - Heatmaps
  - Line Charts
  - Histograms
  - Box Plots
  - Treemaps
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger("ai_pricing.visualizations")


# â”€â”€â”€ Colour Palettes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Consistent colour scheme used across all charts.
COLORS = {
    "primary": "#1a237e",       # Deep indigo
    "secondary": "#0d47a1",     # Blue
    "success": "#2e7d32",       # Green
    "warning": "#f9a825",       # Amber
    "danger": "#c62828",        # Red
    "info": "#00838f",          # Teal
    "purple": "#4a148c",        # Purple
    "grey": "#757575",          # Grey
}

COLOR_PALETTE = px.colors.qualitative.Bold
COLOR_PALETTE_SEQUENTIAL = px.colors.sequential.Blues_r

RISK_COLORS: Dict[str, str] = {
    "Critical": "#d32f2f",
    "High": "#f44336",
    "Medium": "#ff9800",
    "Low": "#4caf50",
}

MARGIN_COLORS: Dict[str, str] = {
    "High": "#2e7d32",
    "Medium": "#1565c0",
    "Low": "#f9a825",
    "Negative": "#c62828",
}

RECOMMENDATION_COLORS: Dict[str, str] = {
    "Increase": "#2e7d32",
    "Decrease": "#c62828",
    "Maintain": "#1565c0",
    "Error": "#757575",
}

INVENTORY_COLORS: Dict[str, str] = {
    "Healthy": "#2e7d32",
    "Low Stock": "#f9a825",
    "Overstocked": "#1565c0",
    "Out of Stock": "#c62828",
    "Critical": "#c62828",
    "Poor": "#f9a825",
    "Fair": "#1565c0",
    "Good": "#2e7d32",
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. KPI METRIC CARDS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_kpi_card(
    label: str,
    value: float,
    prefix: str = "",
    suffix: str = "",
    delta: Optional[float] = None,
    delta_description: str = "",
    inverse_delta: bool = False,
) -> go.Figure:
    """
    Create a single KPI indicator card (gauge-style).

    Args:
        label: Metric label (e.g. "Total Revenue").
        value: Numeric value.
        prefix: Prefix string (e.g. "$").
        suffix: Suffix string (e.g. "%").
        delta: Optional change value to display.
        delta_description: Text for the delta (e.g. "vs last period").
        inverse_delta: If True, positive delta is bad (red).

    Returns:
        Plotly Figure suitable for st.plotly_chart().
    """
    fig = go.Figure()

    display_value = f"{prefix}{value:,.2f}{suffix}" if value else f"{prefix}0{suffix}"

    fig.add_trace(
        go.Indicator(
            mode="number+delta" if delta is not None else "number",
            value=float(value) if value else 0,
            number={
                "font": {"size": 36, "color": COLORS["primary"]},
                "prefix": prefix,
                "suffix": suffix,
                "valueformat": ",.2f" if abs(value if value else 0) >= 10000 else ".2f",
            },
            delta={
                "reference": float(value) - float(delta) if delta else 0,
                "relative": False,
                "valueformat": ".1f",
                "font": {"size": 14},
                "increasing": {"color": COLORS["danger"] if inverse_delta else COLORS["success"]},
                "decreasing": {"color": COLORS["success"] if inverse_delta else COLORS["danger"]},
            } if delta else None,
            title={
                "text": f"<b>{label}</b>",
                "font": {"size": 14, "color": "#424242"},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#424242"},
    )
    return fig


def create_kpi_row(
    df: pd.DataFrame,
    revenue_col: str = "revenue",
    profit_col: str = "expected_profit",
    margin_col: str = "margin_percentage",
    sales_col: str = "sales_volume",
    products_col: str = "product_id",
) -> List[go.Figure]:
    """
    Create a row of 4 KPI cards from a DataFrame.

    Cards: Total Revenue, Total Profit, Avg Margin, Total Products.

    Args:
        df: Analysis DataFrame.
        revenue_col: Column for revenue.
        profit_col: Column for profit.
        margin_col: Column for margin percentage.
        sales_col: Column for sales volume.
        products_col: Column for product count (typically product_id).

    Returns:
        List of 4 Plotly Figures.
    """
    total_revenue = float(df[revenue_col].sum()) if revenue_col in df.columns else 0
    total_profit = float(df[profit_col].sum()) if profit_col in df.columns else 0
    avg_margin = float(df[margin_col].mean()) if margin_col in df.columns else 0
    total_products = int(df[products_col].nunique()) if products_col in df.columns else len(df)

    # Also compute current revenue for delta
    current_revenue = (
        float((df["current_price"] * df["sales_volume"]).sum())
        if "current_price" in df.columns and "sales_volume" in df.columns
        else None
    )

    cards = [
        create_kpi_card(
            "Total Revenue",
            total_revenue,
            prefix="$",
            delta=(total_revenue - current_revenue) if current_revenue else None,
            delta_description="projected change",
        ),
        create_kpi_card(
            "Total Profit",
            total_profit,
            prefix="$",
        ),
        create_kpi_card(
            "Average Margin",
            avg_margin,
            suffix="%",
        ),
        create_kpi_card(
            "Products",
            total_products,
        ),
    ]
    return cards


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. REVENUE & PROFIT CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_revenue_by_category_chart(
    df: pd.DataFrame,
    revenue_col: str = "expected_revenue",
    category_col: str = "category",
    top_n: int = 10,
) -> go.Figure:
    """
    Horizontal bar chart of total revenue by category (top N).

    Args:
        df: DataFrame with revenue and category columns.
        revenue_col: Revenue column name.
        category_col: Category column name.
        top_n: Number of top categories to show.

    Returns:
        Plotly Figure.
    """
    if revenue_col not in df.columns or category_col not in df.columns:
        return _empty_figure("Revenue or category data not available.")

    cat_rev = df.groupby(category_col)[revenue_col].sum().reset_index()
    cat_rev = cat_rev.sort_values(revenue_col, ascending=True).tail(top_n)

    # Colour gradient
    colors = px.colors.sequential.Blues_r[
        :len(cat_rev)
    ] if len(cat_rev) <= 10 else px.colors.sequential.Blues_r

    fig = px.bar(
        cat_rev,
        x=revenue_col,
        y=category_col,
        orientation="h",
        title=f"Top {min(top_n, len(cat_rev))} Categories by Revenue",
        labels={revenue_col: "Revenue ($)", category_col: ""},
        color=revenue_col,
        color_continuous_scale="Blues",
        text_auto=".2s",
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickprefix="$",
        yaxis={"categoryorder": "total ascending"},
        hovermode="y unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_profit_by_category_chart(
    df: pd.DataFrame,
    profit_col: str = "expected_profit",
    category_col: str = "category",
    top_n: int = 10,
) -> go.Figure:
    """
    Horizontal bar chart of total profit by category.

    Args:
        df: DataFrame.
        profit_col: Profit column.
        category_col: Category column.
        top_n: Number of categories.

    Returns:
        Plotly Figure.
    """
    if profit_col not in df.columns or category_col not in df.columns:
        return _empty_figure("Profit or category data not available.")

    cat_profit = df.groupby(category_col)[profit_col].sum().reset_index()
    cat_profit = cat_profit.sort_values(profit_col, ascending=True).tail(top_n)

    # Split positive / negative for colouring
    cat_profit["color"] = cat_profit[profit_col].apply(
        lambda x: COLORS["success"] if x >= 0 else COLORS["danger"]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cat_profit[profit_col],
            y=cat_profit[category_col],
            orientation="h",
            marker_color=cat_profit["color"],
            text=cat_profit[profit_col].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Profit: $%{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Top {min(top_n, len(cat_profit))} Categories by Profit",
        xaxis_title="Profit ($)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickprefix="$",
        yaxis={"categoryorder": "total ascending"},
        hovermode="y unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_revenue_vs_profit_scatter(
    df: pd.DataFrame,
    revenue_col: str = "expected_revenue",
    profit_col: str = "expected_profit",
    category_col: str = "category",
    size_col: str = "sales_volume",
    hover_name_col: str = "product_name",
) -> go.Figure:
    """
    Scatter plot of revenue vs profit, sized by sales volume,
    coloured by category.

    Args:
        df: DataFrame.
        revenue_col: X-axis â€” revenue.
        profit_col: Y-axis â€” profit.
        category_col: Colour encoding.
        size_col: Point size.
        hover_name_col: Hover label.

    Returns:
        Plotly Figure.
    """
    required = [revenue_col, profit_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_figure(f"Missing columns: {missing}")

    fig = px.scatter(
        df,
        x=revenue_col,
        y=profit_col,
        color=category_col if category_col in df.columns else None,
        size=size_col if size_col in df.columns else None,
        hover_name=hover_name_col if hover_name_col in df.columns else None,
        hover_data={
            "product_name": True,
            revenue_col: ":$,.0f",
            profit_col: ":$,.0f",
        },
        title="Revenue vs Profit | Size = Sales Volume",
        labels={
            revenue_col: "Expected Revenue ($)",
            profit_col: "Expected Profit ($)",
        },
        color_discrete_sequence=COLOR_PALETTE,
        opacity=0.75,
        trendline="ols" if len(df) >= 5 else None,
    )

    # Add quadrant lines
    if revenue_col in df.columns and profit_col in df.columns:
        med_rev = df[revenue_col].median()
        med_profit = df[profit_col].median()
        fig.add_vline(x=med_rev, line_dash="dot", line_color="grey", opacity=0.4)
        fig.add_hline(y=med_profit, line_dash="dot", line_color="grey", opacity=0.4)

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. TOP / BOTTOM PRODUCT CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_top_products_chart(
    df: pd.DataFrame,
    value_col: str = "expected_profit",
    name_col: str = "product_name",
    category_col: str = "category",
    top_n: int = 10,
    ascending: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Horizontal bar chart of top or bottom N products by a metric.

    Args:
        df: DataFrame.
        value_col: Metric to rank by.
        name_col: Product name column.
        category_col: Colour by category.
        top_n: Number of products to show.
        ascending: If True, shows bottom N (worst performers).
        title: Custom title (auto-generated if None).

    Returns:
        Plotly Figure.
    """
    if value_col not in df.columns or name_col not in df.columns:
        return _empty_figure(f"Column '{value_col}' or '{name_col}' not found.")

    sorted_df = df.sort_values(value_col, ascending=ascending).head(top_n)

    direction = "Bottom" if ascending else "Top"
    default_title = (
        f"{direction} {top_n} Products by {value_col.replace('_', ' ').title()}"
    )

    fig = px.bar(
        sorted_df,
        x=value_col,
        y=name_col,
        color=category_col if category_col in sorted_df.columns else None,
        orientation="h",
        title=title or default_title,
        labels={
            value_col: value_col.replace("_", " ").title(),
            name_col: "",
        },
        color_discrete_sequence=COLOR_PALETTE,
        text_auto=".2s" if value_col in [
            "expected_revenue", "expected_profit", "revenue", "current_price"
        ] else ".1f",
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x:,.2f}<extra></extra>",
    )
    fig.update_layout(
        height=50 + top_n * 35,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis={"categoryorder": "total ascending"},
        hovermode="y unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_top_profitable_products_chart(
    df: pd.DataFrame,
    profit_col: str = "expected_profit",
    name_col: str = "product_name",
    category_col: str = "category",
    top_n: int = 10,
) -> go.Figure:
    """
    Top N most profitable products (horizontal bar).

    Args:
        df: DataFrame.
        profit_col: Profit column.
        name_col: Product name.
        category_col: Category.
        top_n: Number to show.

    Returns:
        Plotly Figure.
    """
    return create_top_products_chart(
        df,
        value_col=profit_col,
        name_col=name_col,
        category_col=category_col,
        top_n=top_n,
        ascending=False,
        title=f"Top {top_n} Most Profitable Products",
    )


def create_highest_risk_products_chart(
    df: pd.DataFrame,
    risk_col: str = "composite_risk_score",
    name_col: str = "product_name",
    risk_category_col: str = "risk_category",
    top_n: int = 10,
) -> go.Figure:
    """
    Horizontal bar chart of highest risk products.

    Args:
        df: DataFrame.
        risk_col: Risk score column.
        name_col: Product name.
        risk_category_col: Risk category for colour.
        top_n: Number to show.

    Returns:
        Plotly Figure.
    """
    if risk_col not in df.columns:
        return _empty_figure("Risk score data not available.")

    return create_top_products_chart(
        df,
        value_col=risk_col,
        name_col=name_col,
        category_col=risk_category_col if risk_category_col in df.columns else None,
        top_n=top_n,
        ascending=False,
        title=f"Top {top_n} Highest Risk Products",
    )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 4. MARGIN & PRICING CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_margin_distribution_chart(
    df: pd.DataFrame,
    margin_col: str = "margin_percentage",
    category_col: str = "category",
) -> go.Figure:
    """
    Box plot of margin percentage distribution by category.

    Args:
        df: DataFrame.
        margin_col: Margin column (percentage values).
        category_col: Category column.

    Returns:
        Plotly Figure.
    """
    if margin_col not in df.columns:
        return _empty_figure("Margin data not available.")

    fig = px.box(
        df,
        x=category_col if category_col in df.columns else None,
        y=margin_col,
        title="Margin Distribution by Category",
        labels={margin_col: "Margin (%)", category_col: ""},
        color=category_col if category_col in df.columns else None,
        color_discrete_sequence=COLOR_PALETTE,
        points="outliers",
        notched=True,
    )

    fig.add_hline(
        y=float(df[margin_col].median()),
        line_dash="dash",
        line_color="grey",
        annotation_text=f"Median: {df[margin_col].median():.1f}%",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_margin_category_pie(
    df: pd.DataFrame,
    margin_col: str = "margin_percentage",
) -> go.Figure:
    """
    Donut chart of margin category distribution (High / Medium / Low / Negative).

    Uses categorise_margin logic:
      >=40% â†’ High, >=20% â†’ Medium, >=0% â†’ Low, else Negative

    Args:
        df: DataFrame.
        margin_col: Margin percentage column.

    Returns:
        Plotly Figure.
    """
    if margin_col not in df.columns:
        return _empty_figure("Margin data not available.")

    def categorise(val):
        if val >= 40:
            return "High"
        elif val >= 20:
            return "Medium"
        elif val >= 0:
            return "Low"
        return "Negative"

    df_viz = df.copy()
    df_viz["margin_tier"] = df_viz[margin_col].apply(categorise)
    tier_counts = df_viz["margin_tier"].value_counts().reset_index()
    tier_counts.columns = ["tier", "count"]

    # Preserve order
    tier_order = ["High", "Medium", "Low", "Negative"]
    tier_counts["tier"] = pd.Categorical(
        tier_counts["tier"], categories=tier_order, ordered=True
    )
    tier_counts = tier_counts.sort_values("tier")

    fig = px.pie(
        tier_counts,
        values="count",
        names="tier",
        title="Margin Tier Distribution",
        color="tier",
        color_discrete_map=MARGIN_COLORS,
        hole=0.45,
        category_orders={"tier": tier_order},
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_price_vs_optimal_scatter(
    df: pd.DataFrame,
    current_col: str = "current_price",
    optimal_col: str = "optimal_price",
    recommendation_col: str = "recommendation",
    hover_name_col: str = "product_name",
) -> go.Figure:
    """
    Scatter plot comparing current price vs optimal price,
    coloured by recommendation.

    Args:
        df: DataFrame.
        current_col: Current price column.
        optimal_col: Optimal price column.
        recommendation_col: Recommendation column.
        hover_name_col: Product name for hover.

    Returns:
        Plotly Figure.
    """
    if current_col not in df.columns or optimal_col not in df.columns:
        return _empty_figure("Price data not available. Run the pricing engine first.")

    fig = px.scatter(
        df,
        x=current_col,
        y=optimal_col,
        color=recommendation_col if recommendation_col in df.columns else None,
        hover_name=hover_name_col if hover_name_col in df.columns else None,
        hover_data={
            current_col: ":$.2f",
            optimal_col: ":$.2f",
        },
        title="Current Price vs Optimal Price",
        labels={
            current_col: "Current Price ($)",
            optimal_col: "Optimal Price ($)",
        },
        color_discrete_map=RECOMMENDATION_COLORS,
        opacity=0.7,
    )

    # Diagonal reference line
    max_val = max(
        df[current_col].max(), df[optimal_col].max()
    ) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            name="No Change",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_recommendation_pie(
    df: pd.DataFrame,
    recommendation_col: str = "recommendation",
) -> go.Figure:
    """
    Donut chart showing distribution of pricing recommendations.

    Args:
        df: DataFrame.
        recommendation_col: Recommendation column.

    Returns:
        Plotly Figure.
    """
    if recommendation_col not in df.columns:
        return _empty_figure("Recommendation data not available.")

    rec_counts = df[recommendation_col].value_counts().reset_index()
    rec_counts.columns = ["recommendation", "count"]

    # Order
    order = ["Increase", "Decrease", "Maintain"]
    rec_counts["recommendation"] = pd.Categorical(
        rec_counts["recommendation"], categories=order, ordered=True
    )
    rec_counts = rec_counts.sort_values("recommendation")

    fig = px.pie(
        rec_counts,
        values="count",
        names="recommendation",
        title="Pricing Recommendations",
        color="recommendation",
        color_discrete_map=RECOMMENDATION_COLORS,
        hole=0.45,
        category_orders={"recommendation": order},
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 5. RISK CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_risk_heatmap(
    df: pd.DataFrame,
    risk_cols: Optional[List[str]] = None,
    category_col: str = "category",
) -> go.Figure:
    """
    Heatmap of average risk dimension scores by category.

    Args:
        df: DataFrame with risk score columns.
        risk_cols: List of risk dimension columns.
        category_col: Category column.

    Returns:
        Plotly Figure.
    """
    if risk_cols is None:
        risk_cols = [
            "margin_risk_score",
            "demand_risk_score",
            "inventory_risk_score",
            "competitor_risk_score",
        ]

    available = [c for c in risk_cols if c in df.columns]
    if not available or category_col not in df.columns:
        return _empty_figure("Risk dimension data not available.")

    heatmap_data = df.groupby(category_col)[available].mean().round(1)

    # Friendly labels
    label_map = {
        "margin_risk_score": "Margin",
        "demand_risk_score": "Demand",
        "inventory_risk_score": "Inventory",
        "competitive_risk_score": "Competitive",
    }
    heatmap_data = heatmap_data.rename(columns=label_map)

    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        text_auto=".0f",
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="Risk Heatmap by Category",
        labels={"x": "Risk Dimension", "y": "Category", "color": "Score"},
    )

    fig.update_layout(
        height=250 + len(heatmap_data) * 35,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_competitor_price_comparison(
    df: pd.DataFrame,
    our_price_col: str = "current_price",
    competitor_col: str = "competitor_price",
    market_min_col: str = "market_min",
    market_max_col: str = "market_max",
    hover_name_col: str = "product_name",
    top_n: int = 20,
) -> go.Figure:
    """
    Scatter or range chart comparing our price vs competitor price range.

    Shows our price as a marker and competitor range as a thin line.

    Args:
        df: DataFrame.
        our_price_col: Our current price.
        competitor_col: Competitor price (average).
        market_min_col: Market min price.
        market_max_col: Market max price.
        hover_name_col: Product name.
        top_n: Number of products to show (sorted by volume).

    Returns:
        Plotly Figure.
    """
    required = [our_price_col, competitor_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_figure(f"Missing columns: {missing}")

    # Pick top N by revenue / volume
    sort_col = "revenue" if "revenue" in df.columns else "sales_volume"
    sort_col = sort_col if sort_col in df.columns else our_price_col
    df_viz = df.sort_values(sort_col, ascending=False).head(top_n).copy()

    df_viz["price_diff_pct"] = (
        (df_viz[our_price_col] - df_viz[competitor_col])
        / df_viz[competitor_col] * 100
    )

    fig = go.Figure()

    # Competitor range lines (if min/max available)
    has_range = (
        market_min_col in df.columns and market_max_col in df.columns
    )

    for _, row in df_viz.iterrows():
        product = row.get(hover_name_col, "")
        our_price = row[our_price_col]
        comp_price = row[competitor_col]
        diff = row.get("price_diff_pct", 0)

        # Range line from market_min to market_max
        if has_range and pd.notna(row.get(market_min_col)) and pd.notna(row.get(market_max_col)):
            fig.add_trace(
                go.Scatter(
                    x=[row[market_min_col], row[market_max_col]],
                    y=[product, product],
                    mode="lines",
                    line=dict(color="lightgrey", width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Our price markers
    marker_colors = df_viz["price_diff_pct"].apply(
        lambda x: COLORS["danger"] if x > 10
        else (COLORS["success"] if x < -5 else COLORS["primary"])
    )

    fig.add_trace(
        go.Scatter(
            x=df_viz[our_price_col],
            y=df_viz[hover_name_col] if hover_name_col in df_viz.columns else df_viz.index,
            mode="markers",
            marker=dict(
                size=10,
                color=marker_colors,
                line=dict(width=1, color="white"),
                symbol="circle",
            ),
            name="Our Price",
            text=df_viz["price_diff_pct"].apply(lambda x: f"{x:+.1f}% vs competitor"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Our Price: $%{x:.2f}<br>"
                "Diff: %{text}<extra></extra>"
            ),
        )
    )

    # Competitor price markers
    fig.add_trace(
        go.Scatter(
            x=df_viz[competitor_col],
            y=df_viz[hover_name_col] if hover_name_col in df_viz.columns else df_viz.index,
            mode="markers",
            marker=dict(
                size=8,
                color=COLORS["grey"],
                line=dict(width=1, color="white"),
                symbol="diamond",
            ),
            name="Competitor Avg",
            hovertemplate="<b>%{y}</b><br>Competitor: $%{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Our Price vs Competitor (Top {top_n} Products)",
        xaxis_title="Price ($)",
        yaxis_title="",
        height=50 + top_n * 28,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="y unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_inventory_health_chart(
    df: pd.DataFrame,
    health_col: str = "inventory_health",
) -> go.Figure:
    """
    Bar chart of inventory health distribution.

    Args:
        df: DataFrame.
        health_col: Inventory health category column.

    Returns:
        Plotly Figure.
    """
    if health_col not in df.columns:
        return _empty_figure("Inventory health data not available.")

    health_counts = df[health_col].value_counts().reset_index()
    health_counts.columns = ["health", "count"]

    order = ["Good", "Fair", "Poor", "Critical"]
    health_counts["health"] = pd.Categorical(
        health_counts["health"], categories=order, ordered=True
    )
    health_counts = health_counts.sort_values("health")

    fig = px.bar(
        health_counts,
        x="health",
        y="count",
        title="Inventory Health",
        labels={"health": "", "count": "Product Count"},
        color="health",
        color_discrete_map=INVENTORY_COLORS,
        text_auto=True,
        category_orders={"health": order},
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_stock_status_pie(
    df: pd.DataFrame,
    status_col: str = "stock_status",
) -> go.Figure:
    """
    Donut chart of stock status distribution.

    Args:
        df: DataFrame.
        status_col: Stock status column.

    Returns:
        Plotly Figure.
    """
    if status_col not in df.columns:
        return _empty_figure("Stock status data not available.")

    status_counts = df[status_col].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    order = ["Healthy", "Low Stock", "Overstocked", "Out of Stock"]
    status_counts["status"] = pd.Categorical(
        status_counts["status"], categories=order, ordered=True
    )
    status_counts = status_counts.sort_values("status")

    fig = px.pie(
        status_counts,
        values="count",
        names="status",
        title="Stock Status",
        color="status",
        color_discrete_map=INVENTORY_COLORS,
        hole=0.4,
        category_orders={"status": order},
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_days_of_cover_histogram(
    df: pd.DataFrame,
    doc_col: str = "days_of_cover",
    status_col: str = "stock_status",
) -> go.Figure:
    """
    Histogram of days of cover, coloured by stock status.

    Args:
        df: DataFrame.
        doc_col: Days of cover column.
        status_col: Stock status for colour.

    Returns:
        Plotly Figure.
    """
    if doc_col not in df.columns:
        return _empty_figure("Days of cover data not available.")

    # Cap at 365 for visualisation
    df_viz = df.copy()
    df_viz[doc_col] = df_viz[doc_col].clip(upper=365)

    fig = px.histogram(
        df_viz,
        x=doc_col,
        color=status_col if status_col in df_viz.columns else None,
        nbins=25,
        title="Days of Cover Distribution",
        labels={doc_col: "Days of Cover", "count": "Product Count"},
        color_discrete_map=INVENTORY_COLORS,
        barmode="stack",
        opacity=0.85,
    )

    # Threshold lines
    fig.add_vline(
        x=15,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="Low Stock <15d",
    )
    fig.add_vline(
        x=60,
        line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text="Healthy â‰¤60d",
    )
    fig.add_vline(
        x=180,
        line_dash="dash",
        line_color=COLORS["info"],
        annotation_text="Excess >180d",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_inventory_turnover_chart(
    df: pd.DataFrame,
    turnover_col: str = "inventory_turnover",
    category_col: str = "category",
    top_n: int = 15,
) -> go.Figure:
    """
    Horizontal bar of inventory turnover by product (top N).

    Args:
        df: DataFrame.
        turnover_col: Turnover column.
        category_col: Colour by category.
        top_n: Number of products.

    Returns:
        Plotly Figure.
    """
    if turnover_col not in df.columns:
        return _empty_figure("Inventory turnover data not available.")

    name_col = "product_name" if "product_name" in df.columns else df.index
    df_viz = df.nlargest(top_n, turnover_col)

    fig = px.bar(
        df_viz,
        x=turnover_col,
        y=name_col,
        color=category_col if category_col in df_viz.columns else None,
        orientation="h",
        title=f"Top {top_n} Products by Inventory Turnover",
        labels={turnover_col: "Turnover Rate", name_col: ""},
        color_discrete_sequence=COLOR_PALETTE,
        text_auto=".1f",
    )

    fig.update_layout(
        height=50 + top_n * 30,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis={"categoryorder": "total ascending"},
        hovermode="y unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_demand_category_chart(
    df: pd.DataFrame,
    demand_col: str = "demand_category",
    score_col: str = "demand_score",
) -> go.Figure:
    """
    Bar + line combo: demand category counts and average score.

    Args:
        df: DataFrame.
        demand_col: Demand category column.
        score_col: Demand score column.

    Returns:
        Plotly Figure.
    """
    if demand_col not in df.columns:
        return _empty_figure("Demand category data not available.")

    cat_counts = df[demand_col].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]

    order = ["High", "Medium", "Low"]
    cat_counts["category"] = pd.Categorical(
        cat_counts["category"], categories=order, ordered=True
    )
    cat_counts = cat_counts.sort_values("category")

    # Average score per category
    if score_col in df.columns:
        avg_scores = df.groupby(demand_col, observed=False)[score_col].mean().round(2)
        cat_counts["avg_score"] = cat_counts["category"].map(avg_scores)
    else:
        cat_counts["avg_score"] = None

    colours = {"High": "#2e7d32", "Medium": "#f9a825", "Low": "#c62828"}

    fig = go.Figure()

    # Bars
    fig.add_trace(
        go.Bar(
            x=cat_counts["category"],
            y=cat_counts["count"],
            name="Product Count",
            marker_color=[colours.get(c, COLORS["grey"]) for c in cat_counts["category"]],
            text=cat_counts["count"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        )
    )

    # Score line
    if cat_counts["avg_score"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=cat_counts["category"],
                y=cat_counts["avg_score"],
                name="Avg Score",
                mode="lines+markers",
                marker=dict(size=10, color=COLORS["info"]),
                line=dict(color=COLORS["info"], width=3),
                yaxis="y2",
                hovertemplate="Avg Score: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Demand Categories",
        xaxis_title="",
        yaxis_title="Product Count",
        yaxis2=dict(
            title="Avg Score",
            overlaying="y",
            side="right",
            range=[0, 1.05],
        ),
        height=400,
        margin=dict(l=10, r=60, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_demand_forecast_line(
    df: pd.DataFrame,
    forecast_cols: Optional[List[str]] = None,
    product_name_col: str = "product_name",
    top_n: int = 10,
) -> go.Figure:
    """
    Line chart of 4-week demand forecast for top N products.

    Args:
        df: DataFrame with forecast_week_1..4 columns.
        forecast_cols: List of forecast week columns.
        product_name_col: Product name column.
        top_n: Number of products.

    Returns:
        Plotly Figure.
    """
    if forecast_cols is None:
        forecast_cols = [
            "forecast_week_1",
            "forecast_week_2",
            "forecast_week_3",
            "forecast_week_4",
        ]

    available = [c for c in forecast_cols if c in df.columns]
    if not available:
        return _empty_figure("Forecast data not available.")

    sort_col = "forecast_next_30d"
    if sort_col not in df.columns:
        sort_col = available[-1]

    df_viz = df.nlargest(top_n, sort_col) if sort_col in df.columns else df.head(top_n)

    fig = go.Figure()

    for _, row in df_viz.iterrows():
        product = str(row.get(product_name_col, "Unknown"))
        weeks = [f"Week {i+1}" for i in range(len(available))]
        values = [float(row[c]) if pd.notna(row.get(c)) else 0 for c in available]

        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=values,
                mode="lines+markers",
                name=product[:30] + "..." if len(product) > 30 else product,
                line=dict(width=2),
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.0f} units<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"4-Week Demand Forecast (Top {top_n} Products)",
        xaxis_title="Week",
        yaxis_title="Forecasted Units",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_elasticity_scatter(
    df: pd.DataFrame,
    elasticity_col: str = "price_elasticity",
    price_col: str = "current_price",
    demand_col: str = "demand_category",
    volume_col: str = "sales_volume",
) -> go.Figure:
    """
    Scatter plot of price elasticity vs price, coloured by demand category,
    sized by volume.

    Args:
        df: DataFrame.
        elasticity_col: Elasticity column.
        price_col: Price column.
        demand_col: Demand category for colour.
        volume_col: Size encoding.

    Returns:
        Plotly Figure.
    """
    if elasticity_col not in df.columns:
        return _empty_figure("Elasticity data not available.")

    df_viz = df.copy()
    df_viz["abs_elasticity"] = df_viz[elasticity_col].abs()

    fig = px.scatter(
        df_viz,
        x=price_col if price_col in df_viz.columns else df_viz.index,
        y="abs_elasticity",
        color=demand_col if demand_col in df_viz.columns else None,
        size=volume_col if volume_col in df_viz.columns else None,
        hover_name="product_name" if "product_name" in df_viz.columns else None,
        hover_data={
            "product_name": True,
            price_col: ":$.2f",
            "abs_elasticity": ":.2f",
            volume_col: ":,",
        },
        title="Price Elasticity Analysis",
        labels={
            price_col: "Current Price ($)",
            "abs_elasticity": "Price Elasticity (absolute)",
        },
        color_discrete_map={
            "High": "#2e7d32",
            "Medium": "#f9a825",
            "Low": "#c62828",
        },
        opacity=0.7,
    )

    # Reference line at |elasticity| = 1
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text="Elastic (|e|>1)",
    )

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 9. ADDITIONAL UTILITY CHARTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_price_distribution_histogram(
    df: pd.DataFrame,
    price_col: str = "current_price",
    category_col: str = "category",
) -> go.Figure:
    """
    Histogram of price distribution by category.

    Args:
        df: DataFrame.
        price_col: Price column.
        category_col: Category for colour.

    Returns:
        Plotly Figure.
    """
    if price_col not in df.columns:
        return _empty_figure("Price data not available.")

    fig = px.histogram(
        df,
        x=price_col,
        color=category_col if category_col in df.columns else None,
        nbins=25,
        title="Price Distribution",
        labels={price_col: "Price ($)", "count": "Product Count"},
        color_discrete_sequence=COLOR_PALETTE,
        barmode="overlay",
        opacity=0.7,
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_product_treemap(
    df: pd.DataFrame,
    values_col: str = "expected_revenue",
    category_col: str = "category",
    name_col: str = "product_name",
) -> go.Figure:
    """
    Treemap of products, sized by revenue, coloured by category.

    Args:
        df: DataFrame.
        values_col: Size metric.
        category_col: Colour grouping.
        name_col: Label.

    Returns:
        Plotly Figure.
    """
    required = [values_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _empty_figure(f"Column '{missing[0]}' not found.")

    # Cap labels for readability
    df_viz = df.copy()
    if name_col in df_viz.columns:
        df_viz["label"] = df_viz[name_col].astype(str).str[:20]
    else:
        df_viz["label"] = df_viz.index.astype(str)

    fig = px.treemap(
        df_viz,
        path=[category_col, "label"] if category_col in df_viz.columns else ["label"],
        values=values_col,
        color=category_col if category_col in df_viz.columns else None,
        color_discrete_sequence=COLOR_PALETTE,
        title=f"Product Treemap by {values_col.replace('_', ' ').title()}",
        hover_data={
            "label": False,
            values_col: ":$,.0f",
        },
    )

    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>%{value:$,.0f}<extra></extra>",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=5, r=5, t=40, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_alerts_summary_chart(
    alerts: List[Dict[str, Any]],
) -> go.Figure:
    """
    Stacked bar chart of alerts by category and severity.

    Args:
        alerts: List of alert dicts with 'category' and 'severity' keys.

    Returns:
        Plotly Figure.
    """
    if not alerts:
        return _empty_figure("No alerts to display.")

    alert_df = pd.DataFrame(alerts)
    if alert_df.empty:
        return _empty_figure("No alert data.")

    sev_order = ["Critical", "High", "Medium", "Low"]

    pivot = (
        alert_df.groupby(["category", "severity"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        pivot,
        x="category",
        y="count",
        color="severity",
        title="Alerts Summary",
        labels={"category": "Alert Category", "count": "Count"},
        color_discrete_map=RISK_COLORS,
        barmode="stack",
        text_auto=True,
        category_orders={"severity": sev_order},
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _empty_figure(message: str = "No data available.") -> go.Figure:
    """Return an empty figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#757575"),
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
