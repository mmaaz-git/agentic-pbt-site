import pandas as pd
import pandas.api.types as pat

print("Testing pandas_dtype with different Series types:")
print("=" * 50)

# Test with int64 Series
series_int64 = pd.Series([1, 2, 3])
print(f"Series with int64: {series_int64.tolist()}")
print(f"  dtype: {series_int64.dtype}")
try:
    result = pat.pandas_dtype(series_int64)
    print(f"  pat.pandas_dtype(series_int64) = {result}")
except Exception as e:
    print(f"  pat.pandas_dtype(series_int64) raised: {type(e).__name__}: {e}")

print()

# Test with object Series containing None
series_object = pd.Series([None])
print(f"Series with object (None): {series_object.tolist()}")
print(f"  dtype: {series_object.dtype}")

try:
    result = pat.pandas_dtype(series_object.dtype)
    print(f"  pat.pandas_dtype(series_object.dtype) = {result}")
except Exception as e:
    print(f"  pat.pandas_dtype(series_object.dtype) raised: {type(e).__name__}: {e}")

try:
    result = pat.pandas_dtype(series_object)
    print(f"  pat.pandas_dtype(series_object) = {result}")
except Exception as e:
    print(f"  pat.pandas_dtype(series_object) raised: {type(e).__name__}: {e}")

print()

# Test with other object Series
series_mixed = pd.Series(['a', 1, None])
print(f"Series with mixed object: {series_mixed.tolist()}")
print(f"  dtype: {series_mixed.dtype}")

try:
    result = pat.pandas_dtype(series_mixed)
    print(f"  pat.pandas_dtype(series_mixed) = {result}")
except Exception as e:
    print(f"  pat.pandas_dtype(series_mixed) raised: {type(e).__name__}: {e}")

print()

# Test with float Series
series_float = pd.Series([1.0, 2.0, 3.0])
print(f"Series with float64: {series_float.tolist()}")
print(f"  dtype: {series_float.dtype}")
try:
    result = pat.pandas_dtype(series_float)
    print(f"  pat.pandas_dtype(series_float) = {result}")
except Exception as e:
    print(f"  pat.pandas_dtype(series_float) raised: {type(e).__name__}: {e}")