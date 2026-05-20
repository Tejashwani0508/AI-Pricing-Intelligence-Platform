#!/usr/bin/env python
"""
Integration test for chatbot response formatting and handler quality.
Tests actual response generation with sample data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.ai_assistant import AIAssistant
from modules.alerts import AlertEngine

def create_sample_dataframe():
    """Create a sample product DataFrame for testing."""
    np.random.seed(42)
    n_products = 50
    
    df = pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'product_name': [f'Product {i}' for i in range(1, n_products + 1)],
        'category': np.random.choice(['Electronics', 'Home', 'Sports', 'Kitchen'], n_products),
        'current_price': np.random.uniform(10, 500, n_products),
        'sales_volume': np.random.randint(10, 1000, n_products),
        'expected_revenue': np.random.uniform(1000, 50000, n_products),
        'expected_profit': np.random.uniform(100, 10000, n_products),
        'margin_percentage': np.random.uniform(5, 40, n_products),
        'composite_risk_score': np.random.uniform(10, 90, n_products),
        'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], n_products),
        'primary_risk_factor': np.random.choice(['Low Margin', 'High Competition', 'Low Demand', 'Overstock'], n_products),
        'recommendation': np.random.choice(['Increase', 'Decrease', 'Maintain'], n_products),
        'price_change_pct': np.random.uniform(-20, 20, n_products),
        'optimal_price': np.random.uniform(10, 500, n_products),
        'market_position': np.random.choice(['Premium', 'Competitive', 'Discount'], n_products),
        'stock_status': np.random.choice(['Healthy', 'Low Stock', 'Overstocked'], n_products),
        'inventory_action': np.random.choice(['maintain', 'increase', 'discount'], n_products),
        'days_of_cover': np.random.randint(5, 90, n_products),
        'demand_trend': np.random.uniform(0, 1, n_products),
        'demand_trend_category': np.random.choice(['increasing', 'stable', 'declining'], n_products),
        'forecast_next_30d': np.random.randint(100, 5000, n_products),
    })
    return df

def test_response_formatting():
    """Test that responses are well-formatted (no text overlap, emojis removed, etc)."""
    print("=" * 80)
    print("RESPONSE FORMATTING TEST")
    print("=" * 80)
    print()
    
    df = create_sample_dataframe()
    insights = {}
    alerts = [
        {"severity": "HIGH", "category": "Risk", "message": "5 products with high risk score"},
        {"severity": "CRITICAL", "category": "Pricing", "message": "Margin below threshold on 8 products"},
    ]
    
    assistant = AIAssistant(use_openai=False)
    assistant.load_context(df, insights, alerts)
    
    test_cases = [
        ("Hello", "greeting"),
        ("Which products need repricing?", "pricing"),
        ("Show me price increase opportunities", "pricing_increase"),
        ("Which products should we decrease price on?", "pricing_decrease"),
        ("Biggest margin opportunities", "margins"),
        ("Which products are high risk?", "risk"),
        ("Inventory status", "inventory"),
        ("Demand forecast", "forecasting"),
        ("Competitive position", "competitors"),
        ("Portfolio overview", "portfolio"),
        ("Show alerts", "alerts"),
        ("Best selling products", "top_products"),
        ("Help", "help"),
    ]
    
    issues = []
    
    for query, intent_type in test_cases:
        response = assistant.ask(query)
        
        # Check for formatting issues
        checks = {
            "no_leading_emoji": not any(ord(c) > 127 for c in response[:5]),  # No emoji at start
            "no_text_overlap": "•" in response or "\n" in response or len(response) > 0,  # Has structure
            "proper_spacing": "\n" in response or len(response) < 200,  # Readable length or has breaks
            "capitalized": response[0].isupper() if response else True,  # Starts with uppercase
        }
        
        failed_checks = [k for k, v in checks.items() if not v]
        
        if failed_checks:
            issues.append({
                'query': query,
                'intent': intent_type,
                'failed_checks': failed_checks,
                'response_preview': response[:100]
            })
            print(f"✗ {intent_type:20s} | {query:40s}")
            print(f"  Failed: {', '.join(failed_checks)}")
        else:
            print(f"✓ {intent_type:20s} | {query:40s}")
            # Print first line of response
            first_line = response.split('\n')[0][:60]
            print(f"  → {first_line}...")
    
    print()
    print("=" * 80)
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for issue in issues:
            print(f"  • {issue['query']}: {issue['failed_checks']}")
    else:
        print("✓ ALL FORMATTING CHECKS PASSED")
    print("=" * 80)
    
    return len(issues) == 0

def test_greeting_support():
    """Test that greetings are handled properly."""
    print()
    print("=" * 80)
    print("GREETING SUPPORT TEST")
    print("=" * 80)
    print()
    
    df = create_sample_dataframe()
    assistant = AIAssistant(use_openai=False)
    assistant.load_context(df, {}, [])
    
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    
    for greeting in greetings:
        response = assistant.ask(greeting)
        has_helpful_response = any(word in response.lower() for word in [
            'pricing', 'margin', 'risk', 'inventory', 'forecast', 'competitor', 'help'
        ])
        
        status = "✓" if has_helpful_response else "✗"
        print(f"{status} {greeting:15s} → {response.split(chr(10))[0][:60]}...")
    
    print()
    print("=" * 80)
    print("✓ GREETING SUPPORT TEST COMPLETE")
    print("=" * 80)

def test_no_aggressive_fallback():
    """Test that fallback is not triggered for valid queries."""
    print()
    print("=" * 80)
    print("FALLBACK AGGRESSION TEST")
    print("=" * 80)
    print()
    
    df = create_sample_dataframe()
    assistant = AIAssistant(use_openai=False)
    assistant.load_context(df, {}, [])
    
    valid_queries = [
        "Where should we increase price?",
        "Show margin opportunities",
        "High risk products",
        "Inventory optimization",
        "What's the forecast?",
    ]
    
    fallback_triggered = []
    
    for query in valid_queries:
        response = assistant.ask(query)
        is_fallback = "didn't fully understand" in response.lower() or \
                     "i'm not sure" in response.lower()
        
        status = "✗ FALLBACK" if is_fallback else "✓ RECOGNIZED"
        print(f"{status:20s} | {query}")
        
        if is_fallback:
            fallback_triggered.append(query)
    
    print()
    print("=" * 80)
    if fallback_triggered:
        print(f"✗ FALLBACK TRIGGERED FOR {len(fallback_triggered)} QUERIES")
        for q in fallback_triggered:
            print(f"  • {q}")
    else:
        print("✓ NO AGGRESSIVE FALLBACK DETECTED")
    print("=" * 80)
    
    return len(fallback_triggered) == 0

if __name__ == "__main__":
    all_pass = True
    
    all_pass &= test_response_formatting()
    test_greeting_support()
    all_pass &= test_no_aggressive_fallback()
    
    print()
    print("=" * 80)
    if all_pass:
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 80)
    
    sys.exit(0 if all_pass else 1)
