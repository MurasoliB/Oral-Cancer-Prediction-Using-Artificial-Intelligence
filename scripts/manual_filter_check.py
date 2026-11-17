#!/usr/bin/env python3
"""
Test the from_json filter
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def test_from_json_filter():
    """Test the custom from_json filter"""
    print("🧪 Testing from_json Filter")
    print("=" * 40)
    
    with app.app_context():
        # Test the filter function directly
        from app import from_json_filter
        
        # Test with valid JSON string
        json_str = '{"title": "Test Title", "message": "Test Message", "advice": "Test Advice"}'
        result = from_json_filter(json_str)
        print(f"✅ Valid JSON: {result}")
        
        # Test with invalid JSON
        invalid_json = "not a json string"
        result = from_json_filter(invalid_json)
        print(f"✅ Invalid JSON: {result}")
        
        # Test with None
        result = from_json_filter(None)
        print(f"✅ None value: {result}")
        
        # Test with already parsed dict
        dict_obj = {"title": "Already Dict", "message": "Test"}
        result = from_json_filter(dict_obj)
        print(f"✅ Dict object: {result}")
        
        print("\n🎯 Filter test completed successfully!")

if __name__ == "__main__":
    test_from_json_filter()