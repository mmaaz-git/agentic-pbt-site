#!/usr/bin/env python3
"""Test what numpy does with mean of empty arrays"""

import numpy as np
import warnings

print("Testing numpy.mean with empty arrays:")
print("-" * 50)

# Test 1: Empty array
print("\nTest 1: np.mean([])")
with warnings.catch_warnings():
    warnings.simplefilter("always")
    try:
        result = np.mean([])
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        print(f"Is NaN: {np.isnan(result)}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

# Test 2: Empty ndarray
print("\nTest 2: np.mean(np.array([]))")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        result = np.mean(np.array([]))
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        print(f"Is NaN: {np.isnan(result)}")
        if w:
            for warning in w:
                print(f"Warning: {warning.category.__name__}: {warning.message}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")

# Test 3: pandas Series mean
print("\nTest 3: pandas Series mean with empty data")
try:
    import pandas as pd
    s = pd.Series([])
    result = s.mean()
    print(f"pandas Series mean of empty: {result}")
    print(f"Is NaN: {pd.isna(result)}")
except ImportError:
    print("pandas not installed")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test 4: Python statistics module
print("\nTest 4: Python statistics module")
try:
    import statistics
    result = statistics.mean([])
    print(f"Result: {result}")
except statistics.StatisticsError as e:
    print(f"StatisticsError: {e}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")