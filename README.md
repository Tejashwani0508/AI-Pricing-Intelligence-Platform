# AI Pricing Intelligence Platform

An enterprise-grade, multi-product AI pricing system that analyzes hundreds to thousands of products simultaneously using machine learning, statistical modeling, and real-time analytics.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Multi-Product Analysis** | Batch-process 1000s of products with concurrent execution |
| **Smart Pricing Engine** | ML-based price optimization using XGBoost & scikit-learn |
| **Demand Forecasting** | Time-series forecasting with trend & seasonality decomposition |
| **Competitor Analysis** | Track competitor pricing, market position, and share |
| **Inventory Optimization** | Reorder point, safety stock, and turnover analysis |
| **Risk Assessment** | Margin erosion, volatility, and markdown risk scoring |
| **Explainable AI** | SHAP-style feature importance and price decomposition |
| **PDF/Excel Reporting** | Automated report generation with ReportLab & OpenPyXL |
| **Alert System** | Real-time pricing alerts with configurable thresholds |
| **AI Chatbot** | Natural language query assistant for pricing insights |
| **Interactive Dashboard** | Streamlit-based with Plotly visualizations |

---

## 🏗️ Architecture

```
AI_PRICING_PLATFORM/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
├── data/                  # Data storage
│   └── sample_products.csv
├── modules/               # Core business logic
│   ├── csv_processor.py   # Data ingestion & validation
│   ├── pricing_engine.py  # Price optimization
│   ├── demand_forecaster.py
│   ├── competitor_engine.py
│   ├── inventory_engine.py
│   ├── risk_engine.py
│   ├── explainability.py  # AI explainability
│   ├── reporting.py       # PDF/Excel generation
│   └── alerts.py          # Threshold-based alerts
├── dashboard/             # Visualization layer
│   └── visualizations.py
├── chatbot/               # AI assistant
│   └── ai_assistant.py
├── models/                # Trained model storage
├── reports/               # Generated reports output
└── utils/                 # Shared utilities
    ├── __init__.py
    ├── logger.py          # Logging configuration
    ├── config.py          # Configuration management
    └── helpers.py         # Helper functions
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd AI_PRICING_PLATFORM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🎯 Usage

1. **Launch** the app: `streamlit run app.py`
2. **Upload** product data (CSV/Excel) or use the sample dataset
3. **Analyze** pricing, demand, competition, inventory & risk
4. **Optimize** prices with ML-powered recommendations
5. **Generate** PDF/Excel reports
6. **Chat** with the AI assistant for natural-language insights

### Sample Data Columns

| Column | Description |
|--------|-------------|
| `product_id` | Unique product identifier |
| `product_name` | Product display name |
| `category` | Product category |
| `current_price` | Current selling price |
| `cost_price` | Unit cost |
| `competitor_price` | Average competitor price |
| `sales_volume` | Units sold (last 30 days) |
| `inventory_level` | Current stock quantity |
| `demand_trend` | Historical demand signal (0-1) |
| `price_elasticity` | Price sensitivity estimate |

---

## 🧠 Module Overview

### Modules
- **csv_processor.py** — Ingests, validates, and transforms product data
- **pricing_engine.py** — Core ML optimization using XGBoost regression
- **demand_forecaster.py** — Statistical time-series forecasting
- **competitor_engine.py** — Competitive positioning analysis
- **inventory_engine.py** — Inventory health and reorder analytics
- **risk_engine.py** — Multi-factor risk scoring system
- **explainability.py** — Feature contribution & price decomposition
- **reporting.py** — Automated PDF/Excel report generation
- **alerts.py** — Configurable threshold-based alert engine

---

## 📊 Dashboard

The Streamlit dashboard provides six key views:
1. **Overview** — KPIs, summary cards, high-level metrics
2. **Pricing** — Price distribution, optimization recommendations
3. **Demand** — Forecast charts, trend analysis
4. **Competition** — Market position, price comparison
5. **Inventory** — Stock health, reorder alerts
6. **Risk** — Risk heatmap, flagged products

---

## 🤖 AI Chatbot

The built-in AI assistant answers natural-language questions like:
- "Which products have the highest risk?"
- "Show me products where we're priced above competitors"
- "What's the optimal price for product X?"
- "Generate a PDF report for the Electronics category"

> **Note**: For full LLM-powered responses, integrate with OpenAI API by adding your API key to a `.env` file:
> ```
> OPENAI_API_KEY=your-key-here
> ```

---

## 📁 Output

| Artifact | Format | Location |
|----------|--------|----------|
| Pricing Report | PDF | `reports/pricing_report_*.pdf` |
| Data Export | Excel | `reports/pricing_analysis_*.xlsx` |
| Trained Models | .pkl | `models/*.pkl` |
| Charts | Plotly | In-app (downloadable as PNG) |

---

## 🔧 Configuration

Edit `utils/config.py` to customize:
- Risk thresholds
- Alert sensitivity
- Forecasting parameters
- Database connections
- API endpoints

---

## 🚀 Enterprise Expansion

The architecture is designed for scale:
- Add PostgreSQL support via SQLAlchemy
- Extend with FastAPI for REST API layer
- Integrate OpenAI/Claude for advanced NL reasoning
- Deploy on cloud (AWS/GCP/Azure) with Docker
- Add Redis caching for performance
- Implement user authentication & multi-tenancy

---

## 📄 License

MIT License

---

## 🤝 Contributing

Contributions are welcome. Please submit a pull request or open an issue for discussion.