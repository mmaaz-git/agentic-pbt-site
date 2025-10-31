import pandas as pd
import numpy as np
from hypothesis import given, settings
from hypothesis.extra.pandas import series

# Test 1: Hypothesis property-based test
@given(series(dtype=int))
@settings(max_examples=200)
def test_series_abs_non_negative(s):
    result = s.abs()
    assert (result >= 0).all(), f"Found negative value in abs result: {result[result < 0].values}"

# Test 2: Direct reproduction with min_int64
def test_min_int64_overflow():
    min_int64 = np.iinfo(np.int64).min
    s = pd.Series([min_int64])

    print(f"Input value: {s.values[0]}")
    print(f"Min int64: {min_int64}")

    result = s.abs()
    print(f"abs() result: {result.values[0]}")
    print(f"Is result non-negative: {result.values[0] >= 0}")
    print(f"Result type: {type(result.values[0])}")

    # Test with numpy directly
    numpy_result = np.abs(min_int64)
    print(f"\nNumpy abs({min_int64}): {numpy_result}")
    print(f"Is numpy result non-negative: {numpy_result >= 0}")

    # Test with Python's built-in abs
    try:
        python_result = abs(min_int64)
        print(f"\nPython abs({min_int64}): {python_result}")
        print(f"Is Python result non-negative: {python_result >= 0}")
    except Exception as e:
        print(f"\nPython abs() raised: {e}")

    return result.values[0] >= 0

# Test 3: Check other edge cases
def test_edge_cases():
    max_int64 = np.iinfo(np.int64).max
    min_int64 = np.iinfo(np.int64).min

    test_values = [
        min_int64,
        min_int64 + 1,
        -1,
        0,
        1,
        max_int64 - 1,
        max_int64
    ]

    print("\nEdge case testing:")
    for val in test_values:
        s = pd.Series([val], dtype=np.int64)
        result = s.abs()
        print(f"abs({val:20}) = {result.values[0]:20} (non-negative: {result.values[0] >= 0})")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing pandas Series.abs() integer overflow")
    print("=" * 60)

    print("\n1. Direct min_int64 test:")
    print("-" * 40)
    passes_direct = test_min_int64_overflow()

    print("\n2. Edge cases test:")
    print("-" * 40)
    test_edge_cases()

    print("\n3. Hypothesis test (will fail on min_int64):")
    print("-" * 40)
    try:
        test_series_abs_non_negative()
        print("Hypothesis test passed (did not encounter min_int64)")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")

    print("\n" + "=" * 60)
    print(f"Direct test passed: {passes_direct}")