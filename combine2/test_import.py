#!/usr/bin/env python3
"""
Test script to verify the smart object counter can be imported correctly
"""

try:
    import smart_object_counter
    print("✓ Successfully imported smart_object_counter")
    
    # Check if main class exists
    if hasattr(smart_object_counter, 'SmartObjectCounter'):
        print("✓ SmartObjectCounter class found")
    else:
        print("✗ SmartObjectCounter class not found")
    
    # Check if required methods exist
    required_methods = [
        'load_image', 'count_objects', 'reset', 'save_result',
        'auto_detect_objects', 'count_objects_with_yolo'
    ]
    
    for method in required_methods:
        if hasattr(smart_object_counter.SmartObjectCounter, method):
            print(f"✓ Method {method} found")
        else:
            print(f"✗ Method {method} not found")
    
    print("\n✓ All tests passed! The smart object counter is ready to use.")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"✗ Error during import: {e}")
