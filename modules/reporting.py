"""
Automated Reporting Module

Generates professional enterprise reports in three formats:
  - CSV  (portable data export)
  - Excel (formatted multi-sheet workbook with OpenPyXL)
  - PDF  (styled document with ReportLab)

Each report includes:
  - Executive KPIs (revenue, profit, margin, risk)
  - Portfolio summary (category breakdowns)
  - Risk analysis (score distribution, high-risk products)
  - Pricing recommendations (optimization summary)
  - Top/bottom product tables
  - Charts and visual elements

All functions accept a DataFrame and return the file path to the generated report.
"""

import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ─── PDF (ReportLab) ───────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    HRFlowable,
    KeepTogether,
)

# ─── Excel (OpenPyXL) ─────────────────────────────────────────────────────
import openpyxl
from openpyxl.chart import BarChart, PieChart, Reference, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.series import DataPoint
from openpyxl.styles import (
    Font,
    PatternFill,
    Alignment,
    Border,
    Side,
    NamedStyle,
    numbers,
)
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter

from utils.config import AppConfig
from utils.helpers import format_currency, safe_divide

logger = logging.getLogger("ai_pricing.reporting")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & STYLES
# ═══════════════════════════════════════════════════════════════════════════

# Brand colours
BRAND_PRIMARY = "#1a237e"
BRAND_SECONDARY = "#0d47a1"
BRAND_ACCENT = "#2e7d32"
BRAND_WARNING = "#f9a825"
BRAND_DANGER = "#c62828"
BRAND_LIGHT = "#f5f5f5"
BRAND_WHITE = "#ffffff"

# PDF page dimensions
PAGE_WIDTH, PAGE_HEIGHT = letter

# Column mapping for standardised report columns
DEFAULT_KPI_METRICS: List[Dict[str, str]] = [
    {"label": "Total Products", "key": "total_products", "prefix": "", "suffix": ""},
    {"label": "Total Revenue", "key": "total_revenue", "prefix": "$", "suffix": ""},
    {"label": "Total Profit", "key": "total_profit", "prefix": "$", "suffix": ""},
    {"label": "Avg Margin", "key": "avg_margin", "prefix": "", "suffix": "%"},
    {"label": "Avg Risk", "key": "avg_risk", "prefix": "", "suffix": ""},
    {"label": "High Risk", "key": "high_risk_count", "prefix": "", "suffix": ""},
]

RISK_LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ═══════════════════════════════════════════════════════════════════════════
# CSV REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_csv_report(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    index: bool = False,
    **kwargs: Any,
) -> str:
    """
    Generate a CSV report from the analysis DataFrame.

    The CSV includes all analysis columns, making it suitable for
    further processing in Excel, Tableau, or other tools.

    Args:
        df: Analysis DataFrame.
        filename: Output filename (auto-generated if None).
        output_dir: Output directory (default: reports/).
        index: Whether to include the DataFrame index.
        **kwargs: Additional arguments for pd.DataFrame.to_csv().

    Returns:
        Path to the generated CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"pricing_report_{timestamp}.csv"

    config = AppConfig()
    output_path = (
        Path(output_dir) / filename
        if output_dir
        else config.get_report_path(filename)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure numeric formatting is clean
    export_df = df.copy()
    for col in export_df.select_dtypes(include="float").columns:
        export_df[col] = export_df[col].round(4)

    export_df.to_csv(
        output_path,
        index=index,
        encoding="utf-8-sig",
        **kwargs,
    )

    logger.info(f"CSV report generated: {output_path} ({os.path.getsize(output_path):,} bytes)")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════
# EXCEL REPORT  (OpenPyXL)
# ═══════════════════════════════════════════════════════════════════════════

def _excel_header_fill() -> PatternFill:
    return PatternFill(start_color=BRAND_PRIMARY[1:], end_color=BRAND_PRIMARY[1:], fill_type="solid")


def _excel_light_fill() -> PatternFill:
    return PatternFill(start_color="f5f5f5", end_color="f5f5f5", fill_type="solid")


def _excel_green_fill() -> PatternFill:
    return PatternFill(start_color="e8f5e9", end_color="e8f5e9", fill_type="solid")


def _excel_red_fill() -> PatternFill:
    return PatternFill(start_color="ffebee", end_color="ffebee", fill_type="solid")


def _excel_amber_fill() -> PatternFill:
    return PatternFill(start_color="fff8e1", end_color="fff8e1", fill_type="solid")


def _excel_thin_border() -> Border:
    return Border(
        left=Side(style="thin", color="d0d0d0"),
        right=Side(style="thin", color="d0d0d0"),
        top=Side(style="thin", color="d0d0d0"),
        bottom=Side(style="thin", color="d0d0d0"),
    )


def _write_excel_kpi_sheet(ws: Any, kpi_data: Dict[str, Any]) -> None:
    """
    Write a KPI summary sheet with title, metrics table, and formatting.

    Args:
        ws: OpenPyXL worksheet.
        kpi_data: Dict of KPI metrics (e.g. total_products, total_revenue).
    """
    # Title
    ws["A1"] = "AI Pricing Intelligence Platform — Executive Summary"
    ws["A1"].font = Font(bold=True, size=16, color=BRAND_PRIMARY)
    ws.merge_cells("A1:D1")

    ws["A2"] = f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    ws["A2"].font = Font(size=10, color="666666")

    # KPI Table headers
    headers = ["Metric", "Value", "", ""]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=4, column=col_idx, value=header)
        cell.font = Font(bold=True, color=BRAND_WHITE, size=11)
        cell.fill = _excel_header_fill()
        cell.alignment = Alignment(horizontal="center")
        cell.border = _excel_thin_border()

    ws.merge_cells("A4:B4")
    ws.merge_cells("C4:D4")

    # KPI rows
    kpi_rows = [
        ("Products Analyzed", kpi_data.get("total_products", 0)),
        ("Categories", kpi_data.get("total_categories", 0)),
        ("Total Revenue", f"${kpi_data.get('total_revenue', 0):,.2f}"),
        ("Total Profit", f"${kpi_data.get('total_profit', 0):,.2f}"),
        ("Average Margin", f"{kpi_data.get('avg_margin', 0):.2f}%"),
        ("Average Risk Score", f"{kpi_data.get('avg_risk', 0):.1f}"),
        ("High Risk Products", kpi_data.get("high_risk_count", 0)),
        ("Total Stock Value", f"${kpi_data.get('total_stock_value', 0):,.2f}"),
    ]

    for row_idx, (metric, value) in enumerate(kpi_rows, 5):
        ws.cell(row=row_idx, column=1, value=metric).font = Font(size=11)
        ws.cell(row=row_idx, column=1).border = _excel_thin_border()
        ws.cell(row=row_idx, column=2, value=value).font = Font(size=11, bold=True)
        ws.cell(row=row_idx, column=2).border = _excel_thin_border()
        ws.cell(row=row_idx, column=2).alignment = Alignment(horizontal="right")

        if row_idx % 2 == 0:
            ws.cell(row=row_idx, column=1).fill = _excel_light_fill()
            ws.cell(row=row_idx, column=2).fill = _excel_light_fill()

    # Column widths
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 25
    ws.column_dimensions["C"].width = 15
    ws.column_dimensions["D"].width = 15


def _write_excel_dataframe_sheet(
    ws: Any,
    df: pd.DataFrame,
    sheet_title: str = "Data",
    freeze_row: int = 2,
) -> None:
    """
    Write a DataFrame to an Excel worksheet with professional formatting.

    Features:
    - Styled header row (dark background, white text)
    - Alternating row colours
    - Auto-adjusted column widths
    - Frozen header row
    - Number formatting for currency columns

    Args:
        ws: OpenPyXL worksheet.
        df: DataFrame to write.
        sheet_title: Worksheet tab name.
        freeze_row: Row to freeze (default 2 = header row).
    """
    # Detect currency / percentage columns for formatting
    currency_keywords = ["price", "revenue", "profit", "cost", "margin", "stock_value"]
    pct_keywords = ["margin", "change", "pct"]

    # Write header
    for col_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=str(col_name))
        cell.font = Font(bold=True, color=BRAND_WHITE, size=10)
        cell.fill = _excel_header_fill()
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = _excel_thin_border()

    # Write data
    for row_idx, row_data in enumerate(
        dataframe_to_rows(df, index=False, header=False), 2
    ):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font = Font(size=9)
            cell.border = _excel_thin_border()

            # Alternating row colours
            if row_idx % 2 == 0:
                cell.fill = _excel_light_fill()

            # Format detection for column
            col_name_lower = str(df.columns[col_idx - 1]).lower()

            # Currency columns
            if any(kw in col_name_lower for kw in currency_keywords) and not any(
                kw in col_name_lower for kw in pct_keywords
            ):
                if isinstance(value, (int, float)):
                    cell.number_format = '#,##0.00'
                    cell.alignment = Alignment(horizontal="right")

            # Percentage columns
            if any(kw in col_name_lower for kw in pct_keywords):
                if isinstance(value, (int, float)):
                    cell.number_format = '0.00"%"'
                    cell.alignment = Alignment(horizontal="right")

            # Integer columns
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                cell.number_format = '#,##0'
                cell.alignment = Alignment(horizontal="right")

    # Freeze header row
    ws.freeze_panes = f"A{freeze_row}"

    # Auto-adjust column widths
    for col_idx, col_name in enumerate(df.columns, 1):
        max_length = max(
            df[col_name].astype(str).map(len).max() if len(df) > 0 else 0,
            len(str(col_name)),
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = min(
            max(max_length + 3, 10), 35
        )


def _add_excel_category_sheet(wb: Any, df: pd.DataFrame) -> None:
    """
    Add a category analysis sheet with grouped metrics and a bar chart.

    Args:
        wb: OpenPyXL workbook.
        df: Analysis DataFrame.
    """
    if "category" not in df.columns:
        return

    ws = wb.create_sheet("Category Analysis")

    # Aggregate by category
    agg_dict: Dict[str, str] = {
        "product_id": "count",
        "current_price": "mean",
        "profit_margin": "mean",
    }

    if "expected_revenue" in df.columns:
        agg_dict["expected_revenue"] = "sum"
    elif "revenue" in df.columns:
        agg_dict["revenue"] = "sum"
    rev_col = "expected_revenue" if "expected_revenue" in df.columns else "revenue" if "revenue" in df.columns else None

    if "composite_risk_score" in df.columns:
        agg_dict["composite_risk_score"] = "mean"

    cat_data = df.groupby("category").agg(agg_dict).reset_index()
    cat_data.columns = [
        "Category",
        "Product Count",
        "Avg Price",
        "Avg Margin",
    ]
    if rev_col:
        cat_data["Total Revenue"] = (
            df.groupby("category")[rev_col].sum().values
        )
    if "composite_risk_score" in df.columns:
        cat_data["Avg Risk"] = (
            df.groupby("category")["composite_risk_score"].mean().values
        )

    cat_data["Avg Margin"] = cat_data["Avg Margin"].apply(lambda x: round(x * 100, 1))
    cat_data["Avg Price"] = cat_data["Avg Price"].apply(lambda x: round(x, 2))

    _write_excel_dataframe_sheet(ws, cat_data, sheet_title="Category Analysis")

    # Add a bar chart for revenue by category
    if rev_col and "Total Revenue" in cat_data.columns:
        chart = BarChart()
        chart.type = "col"
        chart.title = "Total Revenue by Category"
        chart.y_axis.title = "Revenue ($)"
        chart.x_axis.title = "Category"
        chart.style = 10

        data_ref = Reference(ws, min_col=cat_data.columns.get_loc("Total Revenue") + 1,
                             min_row=1, max_row=len(cat_data) + 1)
        cats_ref = Reference(ws, min_col=1, min_row=2,
                             max_row=len(cat_data) + 1)
        chart.add_data(data_ref, titles_from_data=True)
        chart.set_categories(cats_ref)
        chart.shape = 4

        # Colour the bars
        series = chart.series[0]
        series.graphicalProperties.solidFill = BRAND_SECONDARY

        chart.width = 20
        chart.height = 12
        ws.add_chart(chart, f"E2")


def _add_excel_risk_sheet(wb: Any, df: pd.DataFrame) -> None:
    """
    Add a risk analysis sheet with risk level breakdown.

    Args:
        wb: OpenPyXL workbook.
        df: Analysis DataFrame.
    """
    risk_level_col = "risk_level" if "risk_level" in df.columns else "risk_category"
    if risk_level_col not in df.columns:
        return

    ws = wb.create_sheet("Risk Analysis")

    # Risk level distribution
    level_counts = df[risk_level_col].value_counts()
    risk_data = pd.DataFrame({
        "Risk Level": level_counts.index,
        "Count": level_counts.values,
    })
    risk_data["Percentage"] = (risk_data["Count"] / risk_data["Count"].sum() * 100).round(1)

    # Order
    risk_data["Risk Level"] = pd.Categorical(
        risk_data["Risk Level"],
        categories=RISK_LEVEL_ORDER,
        ordered=True,
    )
    risk_data = risk_data.sort_values("Risk Level")

    _write_excel_dataframe_sheet(ws, risk_data, sheet_title="Risk Analysis")

    # Pie chart for risk distribution
    chart = PieChart()
    chart.title = "Risk Level Distribution"
    chart.style = 10

    data_ref = Reference(ws, min_col=2, min_row=1, max_row=len(risk_data) + 1)
    cats_ref = Reference(ws, min_col=1, min_row=2, max_row=len(risk_data) + 1)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)

    # Pie colours
    colors_seq = [BRAND_DANGER, "#f44336", BRAND_WARNING, BRAND_ACCENT]
    for i, color in enumerate(colors_seq[:len(risk_data)]):
        pt = DataPoint(idx=i)
        pt.graphicalProperties.solidFill = color
        chart.series[0].data_points.append(pt)

    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True
    chart.dataLabels.showCatName = True

    chart.width = 16
    chart.height = 12
    ws.add_chart(chart, f"E2")


def _add_excel_recommendation_sheet(wb: Any, df: pd.DataFrame) -> None:
    """
    Add a pricing recommendations sheet.

    Args:
        wb: OpenPyXL workbook.
        df: Analysis DataFrame.
    """
    if "recommendation" not in df.columns:
        return

    ws = wb.create_sheet("Recommendations")

    rec_counts = df["recommendation"].value_counts().reset_index()
    rec_counts.columns = ["Recommendation", "Count"]

    _write_excel_dataframe_sheet(ws, rec_counts, sheet_title="Recommendations")

    # Bar chart
    chart = BarChart()
    chart.type = "col"
    chart.title = "Pricing Recommendations"
    chart.y_axis.title = "Product Count"

    data_ref = Reference(ws, min_col=2, min_row=1, max_row=len(rec_counts) + 1)
    cats_ref = Reference(ws, min_col=1, min_row=2, max_row=len(rec_counts) + 1)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    chart.style = 10

    colors_map = {
        "Increase": BRAND_ACCENT,
        "Decrease": BRAND_DANGER,
        "Maintain": BRAND_SECONDARY,
    }
    for i in range(len(rec_counts)):
        pt = DataPoint(idx=i)
        rec = rec_counts.iloc[i]["Recommendation"]
        pt.graphicalProperties.solidFill = colors_map.get(rec, "#757575")
        chart.series[0].data_points.append(pt)

    chart.width = 16
    chart.height = 12
    ws.add_chart(chart, f"E2")


def generate_excel_report(
    df: pd.DataFrame,
    insights: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    include_kpi: bool = True,
    include_category: bool = True,
    include_risk: bool = True,
    include_recommendations: bool = True,
    include_full_data: bool = True,
) -> str:
    """
    Generate a professional multi-sheet Excel report.

    Sheets:
    1. Executive Summary — KPIs and high-level metrics
    2. Product Data — full analysed dataset
    3. Category Analysis — grouped metrics + revenue bar chart
    4. Risk Analysis — risk level distribution + pie chart
    5. Pricing Recommendations — recommendation breakdown + bar chart

    Args:
        df: Analysis DataFrame with all computed columns.
        insights: Dict of KPI metrics (auto-computed if None).
        filename: Output filename (auto-generated if None).
        include_kpi: Include the KPI summary sheet.
        include_category: Include category analysis sheet.
        include_risk: Include risk analysis sheet.
        include_recommendations: Include recommendations sheet.
        include_full_data: Include full product data sheet.

    Returns:
        Path to the generated Excel file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"pricing_report_{timestamp}.xlsx"

    config = AppConfig()
    filepath = config.get_report_path(filename)

    # Auto-compute insights if not provided
    if insights is None:
        insights = _compute_insights(df)

    wb = openpyxl.Workbook()

    # Sheet 1: Executive Summary
    if include_kpi:
        ws_summary = wb.active
        ws_summary.title = "Executive Summary"
        _write_excel_kpi_sheet(ws_summary, insights)

    # Sheet 2: Full Product Data
    if include_full_data:
        ws_name = "Product Data" if include_kpi else wb.active
        ws_data = wb.create_sheet("Product Data") if include_kpi else wb.active
        if not include_kpi:
            ws_data.title = "Product Data"
        _write_excel_dataframe_sheet(ws_data, df, sheet_title="Product Data")

    # Sheet 3: Category Analysis
    if include_category:
        _add_excel_category_sheet(wb, df)

    # Sheet 4: Risk Analysis
    if include_risk:
        _add_excel_risk_sheet(wb, df)

    # Sheet 5: Recommendations
    if include_recommendations:
        _add_excel_recommendation_sheet(wb, df)

    wb.save(str(filepath))
    logger.info(
        f"Excel report generated: {filepath} "
        f"({os.path.getsize(filepath):,} bytes, {len(wb.sheetnames)} sheets: {wb.sheetnames})"
    )
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# PDF REPORT  (ReportLab)
# ═══════════════════════════════════════════════════════════════════════════

def _get_pdf_styles() -> Dict[str, ParagraphStyle]:
    """
    Create and return a dictionary of custom paragraph styles for the PDF report.

    Returns:
        Dict of style name -> ParagraphStyle.
    """
    styles = getSampleStyleSheet()

    custom_styles = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=26,
            leading=32,
            spaceAfter=6,
            textColor=colors.HexColor(BRAND_PRIMARY),
            alignment=TA_CENTER,
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=14,
            leading=18,
            spaceAfter=20,
            textColor=colors.HexColor("#455a64"),
            alignment=TA_CENTER,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=16,
            leading=20,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor(BRAND_PRIMARY),
            borderWidth=0,
            borderColor=colors.HexColor(BRAND_PRIMARY),
            borderPadding=4,
        ),
        "KPIValue": ParagraphStyle(
            "KPIValue",
            parent=styles["Normal"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(BRAND_ACCENT),
            alignment=TA_CENTER,
        ),
        "KPILabel": ParagraphStyle(
            "KPILabel",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#616161"),
            alignment=TA_CENTER,
        ),
        "BodyText2": ParagraphStyle(
            "BodyText2",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ),
        "FooterStyle": ParagraphStyle(
            "FooterStyle",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#9e9e9e"),
            alignment=TA_CENTER,
        ),
    }

    # Register with the stylesheet so Paragraph can find them
    for style in custom_styles.values():
        styles.add(style)

    return styles


def _insights_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a standard set of KPI insights from a DataFrame.

    Args:
        df: Analysis DataFrame.

    Returns:
        Dict of KPI metrics.
    """
    insights: Dict[str, Any] = {}

    # Basic counts
    insights["total_products"] = len(df)
    insights["total_categories"] = int(df["category"].nunique()) if "category" in df.columns else 0

    # Revenue
    for col in ["expected_revenue", "revenue"]:
        if col in df.columns:
            insights["total_revenue"] = round(float(df[col].sum()), 2)
            break
    if "total_revenue" not in insights:
        if "current_price" in df.columns and "sales_volume" in df.columns:
            insights["total_revenue"] = round(float((df["current_price"] * df["sales_volume"]).sum()), 2)
        else:
            insights["total_revenue"] = 0

    # Profit
    for col in ["expected_profit"]:
        if col in df.columns:
            insights["total_profit"] = round(float(df[col].sum()), 2)
            break
    if "total_profit" not in insights:
        insights["total_profit"] = 0

    # Margin
    for col in ["margin_percentage", "profit_margin"]:
        if col in df.columns:
            if col == "profit_margin":
                insights["avg_margin"] = round(float(df[col].mean() * 100), 2)
            else:
                insights["avg_margin"] = round(float(df[col].mean()), 2)
            break
    if "avg_margin" not in insights:
        insights["avg_margin"] = 0

    # Risk
    risk_col = "composite_risk_score"
    if risk_col in df.columns:
        insights["avg_risk"] = round(float(df[risk_col].mean()), 1)
        insights["high_risk_count"] = int((df[risk_col] >= 70).sum())
    else:
        insights["avg_risk"] = 0
        insights["high_risk_count"] = 0

    # Inventory
    if "stock_value" in df.columns:
        insights["total_stock_value"] = round(float(df["stock_value"].sum()), 2)

    # Recommendations
    if "recommendation" in df.columns:
        rec_counts = df["recommendation"].value_counts()
        for rec in ["Increase", "Decrease", "Maintain"]:
            insights[f"rec_{rec.lower()}"] = int(rec_counts.get(rec, 0))

    return insights


def _compute_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Alias for _insights_from_dataframe for use in multiple places."""
    return _insights_from_dataframe(df)


def _build_pdf_title_page(story: List[Any], styles: Dict[str, ParagraphStyle],
                           insights: Dict[str, Any]) -> None:
    """
    Build the PDF title page with report name, timestamp, and summary stats.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        insights: KPI insights dict.
    """
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("AI Pricing Intelligence", styles["ReportTitle"]))
    story.append(Paragraph("Platform", styles["ReportTitle"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Automated Pricing Analysis & Recommendations",
        styles["ReportSubtitle"],
    ))
    story.append(Spacer(1, 0.3 * inch))

    # Horizontal rule
    story.append(HRFlowable(
        width="80%",
        thickness=2,
        color=colors.HexColor(BRAND_PRIMARY),
        spaceAfter=12,
        spaceBefore=6,
    ))

    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(
        f"Generated: {timestamp}",
        styles["BodyText2"],
    ))
    story.append(Spacer(1, 0.5 * inch))

    # Quick stats
    stats = [
        f"Products Analyzed: <b>{insights.get('total_products', 'N/A')}</b>",
        f"Categories: <b>{insights.get('total_categories', 'N/A')}</b>",
    ]
    if "total_revenue" in insights:
        stats.append(
            f"Total Revenue: <b>${insights['total_revenue']:,.2f}</b>"
        )
    if "high_risk_count" in insights:
        stats.append(
            f"High Risk Products: <b>{insights['high_risk_count']}</b>"
        )

    for stat in stats:
        story.append(Paragraph(stat, styles["BodyText2"]))


def _build_pdf_kpi_section(story: List[Any], styles: Dict[str, ParagraphStyle],
                            insights: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Build the KPI summary section with a 2x3 grid of metric cards.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        insights: KPI insights dict.
        df: DataFrame (used for row count fallback).
    """
    story.append(Paragraph("Executive KPIs", styles["SectionHeader"]))

    kpi_data = [
        ("Products", str(insights.get("total_products", len(df)))),
        ("Categories", str(insights.get("total_categories", "N/A"))),
        (
            "Avg Margin",
            f"{insights.get('avg_margin', 0):.1f}%",
        ),
        (
            "Total Revenue",
            f"${insights.get('total_revenue', 0):,.0f}",
        ),
        ("High Risk", str(insights.get("high_risk_count", 0))),
        (
            "Avg Risk",
            f"{insights.get('avg_risk', 0):.1f}",
        ),
    ]

    # 2 rows x 3 columns grid
    for row_start in range(0, 6, 3):
        row_kpis = kpi_data[row_start:row_start + 3]
        row_cells = []
        for label, value in row_kpis:
            cell_text = (
                f"<b>{value}</b><br/>"
                f"<font size='8' color='#616161'>{label}</font>"
            )
            row_cells.append(Paragraph(cell_text, styles["Normal"]))

        kpi_table = Table(
            [row_cells],
            colWidths=[2.0 * inch] * len(row_cells),
        )
        kpi_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, 0), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.1 * inch))


def _build_pdf_category_table(story: List[Any], styles: Dict[str, ParagraphStyle],
                                df: pd.DataFrame) -> None:
    """
    Build the category summary table section.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        df: DataFrame.
    """
    if "category" not in df.columns:
        return

    story.append(Paragraph("Portfolio Summary by Category", styles["SectionHeader"]))

    # Aggregate
    agg_cols: Dict[str, str] = {"product_id": "count"}
    for col in ["current_price", "profit_margin", "revenue", "expected_revenue",
                "composite_risk_score"]:
        if col in df.columns:
            agg_cols[col] = "mean" if col != "revenue" and col != "expected_revenue" else "sum"

    cat_data = df.groupby("category").agg(agg_cols).reset_index()

    # Build display columns
    display_cols = ["category", "product_id", "current_price", "profit_margin"]
    available_display = [c for c in display_cols if c in cat_data.columns]
    if "revenue" in cat_data.columns or "expected_revenue" in cat_data.columns:
        rev_col = "expected_revenue" if "expected_revenue" in cat_data.columns else "revenue"
        available_display.append(rev_col)
    if "composite_risk_score" in cat_data.columns:
        available_display.append("composite_risk_score")

    # Build display DataFrame with clean column labels
    display_df = cat_data[available_display].copy()
    col_labels = {"category": "Category", "product_id": "Products",
                  "current_price": "Avg Price", "profit_margin": "Avg Margin",
                  "composite_risk_score": "Avg Risk"}
    if "expected_revenue" in available_display:
        col_labels["expected_revenue"] = "Total Revenue"
    elif "revenue" in available_display:
        col_labels["revenue"] = "Total Revenue"
    display_df = display_df.rename(columns=col_labels)

    # Format
    if "Avg Price" in display_df.columns:
        display_df["Avg Price"] = display_df["Avg Price"].apply(lambda x: f"${x:.2f}")
    if "Avg Margin" in display_df.columns:
        display_df["Avg Margin"] = display_df["Avg Margin"].apply(lambda x: f"{x*100:.1f}%")
    if "Total Revenue" in display_df.columns:
        display_df["Total Revenue"] = display_df["Total Revenue"].apply(lambda x: f"${x:,.0f}")
    if "Avg Risk" in display_df.columns:
        display_df["Avg Risk"] = display_df["Avg Risk"].apply(lambda x: f"{x:.1f}")

    # Build PDF table
    table_data = [list(display_df.columns)] + display_df.values.tolist()

    if len(table_data) > 1:
        col_w = min(1.2 * inch, (6.5 * inch) / len(display_df.columns))
        cat_table = Table(table_data, colWidths=[col_w] * len(display_df.columns))

        cat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(cat_table)


def _build_pdf_risk_section(story: List[Any], styles: Dict[str, ParagraphStyle],
                              df: pd.DataFrame) -> None:
    """
    Build the risk assessment section with level distribution table.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        df: DataFrame.
    """
    risk_col = "risk_level" if "risk_level" in df.columns else "risk_category"
    score_col = "composite_risk_score"

    if risk_col not in df.columns:
        return

    story.append(Paragraph("Risk Analysis", styles["SectionHeader"]))

    # Risk level distribution
    level_counts = df[risk_col].value_counts()
    risk_data = [
        ["Risk Level", "Count", "Percentage", "Avg Score"]
    ]

    for level in RISK_LEVEL_ORDER:
        if level in level_counts.index:
            count = int(level_counts[level])
            pct = count / len(df) * 100
            avg = (
                round(float(df[df[risk_col] == level][score_col].mean()), 1)
                if score_col in df.columns
                else 0
            )
            risk_data.append([level, str(count), f"{pct:.1f}%", str(avg)])

    if len(risk_data) > 1:
        risk_table = Table(risk_data, colWidths=[1.5 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_DANGER)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(risk_table)

    # Top high-risk products
    if score_col in df.columns:
        high_risk = df[df[score_col] >= 50].sort_values(score_col, ascending=False).head(5)
        if not high_risk.empty:
            story.append(Spacer(1, 0.15 * inch))
            story.append(Paragraph("Top High-Risk Products", styles["SectionHeader"]))

            hr_data = [["Product", "Risk Score", "Risk Level", "Primary Factor"]]
            for _, row in high_risk.iterrows():
                hr_data.append([
                    str(row.get("product_name", "Unknown"))[:30],
                    str(round(row.get(score_col, 0), 1)),
                    str(row.get(risk_col, "")),
                    str(row.get("primary_risk_factor", "")),
                ])

            hr_table = Table(hr_data, colWidths=[2.0 * inch, 1.0 * inch, 1.0 * inch, 1.5 * inch])
            hr_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_DANGER)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ]))
            story.append(hr_table)


def _build_pdf_recommendation_section(story: List[Any], styles: Dict[str, ParagraphStyle],
                                       df: pd.DataFrame) -> None:
    """
    Build the pricing recommendations section.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        df: DataFrame.
    """
    if "recommendation" not in df.columns:
        return

    story.append(Paragraph("Pricing Recommendations", styles["SectionHeader"]))

    rec_counts = df["recommendation"].value_counts()
    rec_data = [["Recommendation", "Count", "Percentage"]]
    for rec in ["Increase", "Decrease", "Maintain"]:
        if rec in rec_counts.index:
            count = int(rec_counts[rec])
            pct = count / len(df) * 100
            rec_data.append([rec, str(count), f"{pct:.1f}%"])

    if len(rec_data) > 1:
        rec_table = Table(rec_data, colWidths=[2.0 * inch, 1.0 * inch, 1.5 * inch])
        rec_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(rec_table)

    # Top products to increase
    inc_df = df[df["recommendation"] == "Increase"].nlargest(
        5, "price_change_pct" if "price_change_pct" in df.columns else "current_price"
    )
    if not inc_df.empty:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Top Candidates for Price Increase", styles["SectionHeader"]))
        inc_data = [["Product", "Current", "Optimal", "Change"]]
        for _, row in inc_df.iterrows():
            inc_data.append([
                str(row.get("product_name", "Unknown"))[:30],
                f"${row.get('current_price', 0):.2f}",
                f"${row.get('optimal_price', 0):.2f}",
                f"{row.get('price_change_pct', 0):+.1f}%",
            ])
        inc_table = Table(inc_data, colWidths=[2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
        inc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_ACCENT)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(inc_table)


def _build_pdf_top_products(story: List[Any], styles: Dict[str, ParagraphStyle],
                              df: pd.DataFrame) -> None:
    """
    Build a top products by revenue table.

    Args:
        story: ReportLab story list.
        styles: Paragraph style dict.
        df: DataFrame.
    """
    sort_col = "expected_revenue" if "expected_revenue" in df.columns else "revenue" if "revenue" in df.columns else "current_price"
    if sort_col not in df.columns:
        return

    story.append(Paragraph("Top Products by Revenue", styles["SectionHeader"]))

    top_n = min(10, len(df))
    top_df = df.nlargest(top_n, sort_col)

    cols = ["product_name", "category", "current_price"]
    if "expected_revenue" in df.columns:
        cols.append("expected_revenue")
    elif "revenue" in df.columns:
        cols.append("revenue")
    if "margin_percentage" in df.columns:
        cols.append("margin_percentage")
    elif "profit_margin" in df.columns:
        cols.append("profit_margin")

    available = [c for c in cols if c in top_df.columns]
    if len(available) < 2:
        return

    display_df = top_df[available].copy()
    col_labels = {
        "product_name": "Product", "category": "Category",
        "current_price": "Price", "expected_revenue": "Revenue",
        "revenue": "Revenue", "margin_percentage": "Margin",
        "profit_margin": "Margin",
    }
    display_df = display_df.rename(columns=col_labels)

    if "Price" in display_df.columns:
        display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
    if "Revenue" in display_df.columns:
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"${x:,.0f}")
    if "Margin" in display_df.columns:
        margin_col = [c for c in display_df.columns if c == "Margin"][0]
        if "profit_margin" in available:
            display_df["Margin"] = display_df["Margin"].apply(lambda x: f"{x*100:.1f}%")
        else:
            display_df["Margin"] = display_df["Margin"].apply(lambda x: f"{x:.1f}%")

    table_data = [list(display_df.columns)] + display_df.values.tolist()
    col_w = min(1.3 * inch, (6.5 * inch) / len(display_df.columns))

    prod_table = Table(table_data, colWidths=[col_w] * len(display_df.columns))
    prod_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BRAND_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    story.append(prod_table)


def generate_pdf_report(
    df: pd.DataFrame,
    insights: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive, professionally styled PDF report.

    Sections:
    1. Title page with executive summary stats
    2. Key Performance Indicators (2x3 grid)
    3. Portfolio Summary by Category
    4. Risk Analysis (level distribution + top high-risk products)
    5. Pricing Recommendations (distribution + top increase candidates)
    6. Top Products by Revenue

    Args:
        df: Analysis DataFrame with all computed columns.
        insights: Dict of KPI metrics (auto-computed if None).
        filename: Output filename (auto-generated if None).

    Returns:
        Path to the generated PDF file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"pricing_report_{timestamp}.pdf"

    config = AppConfig()
    filepath = config.get_report_path(filename)

    styles = _get_pdf_styles()
    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        title="AI Pricing Intelligence Report",
        author="AI Pricing Platform",
    )

    if insights is None:
        insights = _compute_insights(df)

    story: List[Any] = []

    # Page 1: Title
    _build_pdf_title_page(story, styles, insights)
    story.append(PageBreak())

    # Page 2: KPIs + Category Summary
    _build_pdf_kpi_section(story, styles, insights, df)
    story.append(Spacer(1, 0.2 * inch))
    _build_pdf_category_table(story, styles, df)
    story.append(PageBreak())

    # Page 3: Risk + Recommendations
    _build_pdf_risk_section(story, styles, df)
    story.append(Spacer(1, 0.2 * inch))
    _build_pdf_recommendation_section(story, styles, df)
    story.append(PageBreak())

    # Page 4: Top Products
    _build_pdf_top_products(story, styles, df)

    doc.build(story)
    logger.info(
        f"PDF report generated: {filepath} "
        f"({os.path.getsize(filepath):,} bytes)"
    )
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════
# CLASS-BASED API  (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Enterprise report generator for the AI Pricing Intelligence Platform.

    Generates reports in three formats:
    - CSV  (portable data)
    - Excel (multi-sheet workbook with charts, OpenPyXL)
    - PDF  (styled document, ReportLab)

    Each report includes KPIs, portfolio summary, risk analysis,
    pricing recommendations, and top product tables.

    Usage:
        generator = ReportGenerator()
        path = generator.generate_pdf_report(df)
        path = generator.generate_excel_report(df)
        path = generator.generate_csv_report(df)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the report generator.

        Args:
            config: Application configuration.
        """
        self.config = config or AppConfig()

    def generate_csv_report(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a CSV report from the analysis DataFrame.

        Args:
            df: Analysis DataFrame.
            filename: Output filename.
            **kwargs: Additional CSV args.

        Returns:
            Path to generated CSV file.
        """
        return generate_csv_report(df, filename=filename, **kwargs)

    def generate_excel_report(
        self,
        df: pd.DataFrame,
        insights: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a formatted multi-sheet Excel report.

        Args:
            df: Analysis DataFrame.
            insights: KPI metrics dict.
            filename: Output filename.

        Returns:
            Path to generated Excel file.
        """
        return generate_excel_report(df, insights=insights, filename=filename)

    def generate_pdf_report(
        self,
        df: pd.DataFrame,
        insights: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a professionally styled PDF report.

        Args:
            df: Analysis DataFrame.
            insights: KPI metrics dict.
            filename: Output filename.

        Returns:
            Path to generated PDF file.
        """
        return generate_pdf_report(df, insights=insights, filename=filename)

    def generate_risk_report(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
    ) -> str:
        """
        Generate a focused risk assessment Excel report.

        Args:
            df: DataFrame with risk analysis columns.
            filename: Output filename.

        Returns:
            Path to generated Excel file.
        """
        risk_cols = [
            "product_name", "category",
            "composite_risk_score", "risk_level",
            "primary_risk_factor",
            "demand_risk_score", "profitability_risk_score",
            "inventory_risk_score", "competitor_risk_score",
            "margin_risk_score",
        ]
        available = [c for c in risk_cols if c in df.columns]

        risk_df = df[available].sort_values(
            "composite_risk_score", ascending=False
        )

        insights = _compute_insights(risk_df)
        insights["report_type"] = "Risk Assessment"
        insights["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_report_{timestamp}.xlsx"

        return self.generate_excel_report(
            risk_df, insights=insights, filename=filename
        )