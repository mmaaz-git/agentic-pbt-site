#!/usr/bin/env python3
import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st

# Test the basic reproduction case
print("=" * 60)
print("Basic Reproduction Test")
print("=" * 60)

series = pd.Series([31536001.0])
print(f"Original: {series.iloc[0]} (dtype: {series.dtype})")

json_str = series.to_json(orient="split")
print(f"JSON string: {json_str}")

result = pd.read_json(StringIO(json_str), typ="series", orient="split")
print(f"After round-trip: {result.iloc[0]} (dtype: {result.dtype})")

print("\n" + "=" * 60)
print("Additional Examples")
print("=" * 60)

# Test edge case at threshold
series1 = pd.Series([31536000.0])
json_str1 = series1.to_json(orient="split")
result1 = pd.read_json(StringIO(json_str1), typ="series", orient="split")
print(f"\nValue 31536000.0:")
print(f"  Original: {series1.iloc[0]} (dtype: {series1.dtype})")
print(f"  After round-trip: {result1.iloc[0]} (dtype: {result1.dtype})")

# Test larger value
series2 = pd.Series([1000000000.5])
json_str2 = series2.to_json(orient="split")
result2 = pd.read_json(StringIO(json_str2), typ="series", orient="split")
print(f"\nValue 1000000000.5:")
print(f"  Original: {series2.iloc[0]} (dtype: {series2.dtype})")
print(f"  After round-trip: {result2.iloc[0]} (dtype: {result2.dtype})")

# Test below threshold
series3 = pd.Series([31535999.0])
json_str3 = series3.to_json(orient="split")
result3 = pd.read_json(StringIO(json_str3), typ="series", orient="split")
print(f"\nValue 31535999.0 (below threshold):")
print(f"  Original: {series3.iloc[0]} (dtype: {series3.dtype})")
print(f"  After round-trip: {result3.iloc[0]} (dtype: {result3.dtype})")

# Test with convert_dates=False workaround
print("\n" + "=" * 60)
print("Workaround Test (convert_dates=False)")
print("=" * 60)

series4 = pd.Series([31536001.0])
json_str4 = series4.to_json(orient="split")
result4 = pd.read_json(StringIO(json_str4), typ="series", orient="split", convert_dates=False)
print(f"Original: {series4.iloc[0]} (dtype: {series4.dtype})")
print(f"After round-trip with convert_dates=False: {result4.iloc[0]} (dtype: {result4.dtype})")

# Run the hypothesis test
print("\n" + "=" * 60)
print("Running Hypothesis Test")
print("=" * 60)

@given(st.floats(min_value=31536001, max_value=1e10, allow_nan=False, allow_infinity=False))
def test_series_json_roundtrip_preserves_dtype_and_value(value):
    series = pd.Series([value])
    json_str = series.to_json(orient="split")
    result = pd.read_json(StringIO(json_str), typ="series", orient="split")

    assert series.dtype == result.dtype, f"dtype changed from {series.dtype} to {result.dtype} for value {value}"
    assert series.iloc[0] == result.iloc[0], f"value changed from {series.iloc[0]} to {result.iloc[0]}"

try:
    # Run a few test examples
    test_series_json_roundtrip_preserves_dtype_and_value()
    print("Hypothesis test completed - checking multiple values")
except AssertionError as e:
    print(f"Hypothesis test failed with error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Check the specific threshold value that pandas uses
print("\n" + "=" * 60)
print("Checking threshold behavior")
print("=" * 60)

# Test values around the threshold
test_values = [31535999, 31536000, 31536001, 31536000.5, 86400000]
for val in test_values:
    s = pd.Series([float(val)])
    j = s.to_json(orient="split")
    r = pd.read_json(StringIO(j), typ="series", orient="split")
    print(f"Value {val}: {s.dtype} -> {r.dtype}, {s.iloc[0]} -> {r.iloc[0]}")