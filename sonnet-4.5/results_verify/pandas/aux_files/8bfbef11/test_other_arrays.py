import pandas as pd
import numpy as np

print("Testing take([]) with different ExtensionArray types:")

# 1. String array
print("\n1. StringArray:")
try:
    str_arr = pd.array(['a', 'b', 'c'], dtype=pd.StringDtype())
    print(f"   Array: {str_arr}")
    result = str_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. Categorical array
print("\n2. Categorical:")
try:
    cat_arr = pd.array(['a', 'b', 'c'], dtype='category')
    print(f"   Array: {cat_arr}")
    result = cat_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Period array
print("\n3. PeriodArray:")
try:
    period_arr = pd.array(pd.period_range('2020-01', periods=3, freq='M'))
    print(f"   Array: {period_arr}")
    result = period_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 4. Interval array
print("\n4. IntervalArray:")
try:
    interval_arr = pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    print(f"   Array: {interval_arr}")
    result = interval_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 5. Boolean array
print("\n5. BooleanArray:")
try:
    bool_arr = pd.array([True, False, True], dtype='boolean')
    print(f"   Array: {bool_arr}")
    result = bool_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 6. Arrow string array
print("\n6. ArrowStringArray:")
try:
    arrow_str_arr = pd.array(['a', 'b', 'c'], dtype='string[pyarrow]')
    print(f"   Array: {arrow_str_arr}")
    result = arrow_str_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")

# 7. Arrow float array
print("\n7. ArrowFloatArray:")
try:
    arrow_float_arr = pd.array([1.0, 2.0, 3.0], dtype='float64[pyarrow]')
    print(f"   Array: {arrow_float_arr}")
    result = arrow_float_arr.take([])
    print(f"   take([]): {result}, length: {len(result)}, dtype: {result.dtype}")
except Exception as e:
    print(f"   ERROR: {e}")