import numpy as np
import pandas as pd
import warnings

print("Testing NumPy and pandas behavior with insufficient data for ddof")
print("=" * 60)

# Test with NumPy
print("\nNumPy tests:")
print("-" * 40)

data = np.array([5.0])
print(f"Data: {data}")
print(f"Number of elements: {len(data)}")

print("\nNumPy var with ddof=1 (sample variance):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        result = np.var(data, ddof=1)
        print(f"Result: {result}")
        if w:
            for warning in w:
                print(f"Warning: {warning.category.__name__}: {warning.message}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

print("\nNumPy var with ddof=0 (population variance):")
try:
    result = np.var(data, ddof=0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with 2 elements and ddof=2
data2 = np.array([1.0, 2.0])
print(f"\nData: {data2}")
print(f"Number of elements: {len(data2)}")
print("NumPy var with ddof=2:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        result = np.var(data2, ddof=2)
        print(f"Result: {result}")
        if w:
            for warning in w:
                print(f"Warning: {warning.category.__name__}: {warning.message}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# Test with pandas
print("\n" + "=" * 60)
print("\npandas tests:")
print("-" * 40)

s = pd.Series([5.0])
print(f"Data: {s.values}")
print(f"Number of elements: {len(s)}")

print("\npandas var with ddof=1 (default):")
try:
    result = s.var()  # pandas uses ddof=1 by default
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\npandas var with ddof=0:")
try:
    result = s.var(ddof=0)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with 2 elements and ddof=2
s2 = pd.Series([1.0, 2.0])
print(f"\nData: {s2.values}")
print(f"Number of elements: {len(s2)}")
print("pandas var with ddof=2:")
try:
    result = s2.var(ddof=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Additional test - what happens with an empty array?
print("\n" + "=" * 60)
print("\nEdge case - empty arrays:")
print("-" * 40)

empty = np.array([])
print("NumPy var on empty array with ddof=0:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        result = np.var(empty, ddof=0)
        print(f"Result: {result}")
        if w:
            for warning in w:
                print(f"Warning: {warning.category.__name__}: {warning.message}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

empty_series = pd.Series([])
print("\npandas var on empty series:")
try:
    result = empty_series.var()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")