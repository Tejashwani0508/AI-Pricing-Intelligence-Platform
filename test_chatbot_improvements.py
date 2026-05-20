#!/usr/bin/env python
"""
Test script to verify chatbot improvements.
Tests intent classification with the user's example queries.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.ai_assistant import IntentClassifier

# Test queries from user requirements
TEST_QUERIES = [
    # Pricing intent examples
    ("Where should we increase price?", "PRICING_INCREASE"),
    ("Which products need price increases?", "PRICING_INCREASE"),
    ("Show products to raise prices", "PRICING_INCREASE"),
    ("What should be repriced upward?", "PRICING_INCREASE"),
    ("Which items are underpriced?", "PRICING_INCREASE"),
    
    # Price decrease
    ("Where should we decrease price?", "PRICING_DECREASE"),
    ("Which products need markdowns?", "PRICING_DECREASE"),
    ("Show overpriced products", "PRICING_DECREASE"),
    ("Price reduction opportunities", "PRICING_DECREASE"),
    
    # Margins
    ("Biggest margin opportunities", "MARGINS"),
    ("Margin improvement opportunities", "MARGINS"),
    ("Which products can improve profit?", "MARGINS"),
    ("Low margin products", "MARGINS"),
    ("Profit optimization suggestions", "MARGINS"),
    
    # Risk
    ("Which products are high risk?", "RISK"),
    ("Show risky products", "RISK"),
    ("Risk alerts", "RISK"),
    ("Products needing attention", "RISK"),
    ("Highest risk items", "RISK"),
    
    # Inventory
    ("Inventory problems", "INVENTORY"),
    ("Overstock products", "INVENTORY"),
    ("Excess inventory", "INVENTORY"),
    ("Low stock items", "INVENTORY"),
    ("Inventory optimization opportunities", "INVENTORY"),
    
    # Forecasting
    ("Demand forecast", "FORECASTING"),
    ("Future demand trends", "FORECASTING"),
    ("Sales prediction", "FORECASTING"),
    ("Expected demand", "FORECASTING"),
    
    # Competitors
    ("Competitor pricing", "COMPETITORS"),
    ("Market comparison", "COMPETITORS"),
    ("Pricing gap analysis", "COMPETITORS"),
    ("Competitive position", "COMPETITORS"),
    
    # Greetings
    ("hi", "GREETING"),
    ("hello", "GREETING"),
    ("hey", "GREETING"),
    ("good morning", "GREETING"),
    ("good evening", "GREETING"),
    
    # Portfolio/Overview
    ("Portfolio summary", "PORTFOLIO"),
    ("Executive summary", "PORTFOLIO"),
    ("What are key insights?", "PORTFOLIO"),
    
    # Top products
    ("Best selling products", "TOP_PRODUCTS"),
    ("Top revenue products", "TOP_PRODUCTS"),
    
    # Help
    ("Help", "HELP"),
    ("What can you do?", "HELP"),
    ("What can I ask?", "HELP"),
]

def test_intent_classification():
    """Test intent classification on all user examples."""
    print("=" * 80)
    print("CHATBOT INTENT CLASSIFICATION TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for query, expected_intent in TEST_QUERIES:
        detected_intent, confidence = IntentClassifier.classify(query.lower())
        
        match = detected_intent == expected_intent
        status = "✓ PASS" if match else "✗ FAIL"
        
        print(f"{status} | {query:50s} | Expected: {expected_intent:20s} | Got: {detected_intent:20s} ({confidence:.2f})")
        
        if match:
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(TEST_QUERIES)} tests")
    print(f"Success Rate: {(passed/len(TEST_QUERIES)*100):.1f}%")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = test_intent_classification()
    sys.exit(0 if success else 1)
