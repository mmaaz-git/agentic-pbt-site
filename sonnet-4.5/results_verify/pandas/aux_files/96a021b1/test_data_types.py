import pandas as pd
import pandas.util
import numpy as np

# Test different data types
test_cases = [
    ("int64 values", pd.Series([1, 2, 3])),
    ("float values", pd.Series([1.0, 2.0, 3.0])),
    ("string values", pd.Series(["a", "b", "c"])),
    ("object dtype (large int)", pd.Series([-9_223_372_036_854_775_809])),
    ("mixed types", pd.Series([1, "a", 3.0])),
    ("bool values", pd.Series([True, False, True])),
    ("datetime", pd.Series(pd.date_range('2021-01-01', periods=3))),
]

for desc, series in test_cases:
    print(f"\n{desc} (dtype: {series.dtype}):")
    try:
        result = pandas.util.hash_pandas_object(series, hash_key="test")
        print(f"  Success: {result.iloc[0]}")
    except ValueError as e:
        print(f"  Error: {e}")
    except Exception as e:
        print(f"  Other error: {e}")