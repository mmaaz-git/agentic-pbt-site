"""Test to reproduce the sorted_columns bug with None values"""

# First test the failing case from the bug report
from dask.dataframe.io.parquet.core import sorted_columns

print("Testing the exact failing case from bug report:")
stats = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 5, "max": 10}]},
]

try:
    result = sorted_columns(stats)
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError occurred: {e}")
except Exception as e:
    print(f"Other error: {e}")

print("\nTesting another variant with same issue:")
stats2 = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 0, "max": 0}]},
]

try:
    result2 = sorted_columns(stats2)
    print(f"Result: {result2}")
except TypeError as e:
    print(f"TypeError occurred: {e}")
except Exception as e:
    print(f"Other error: {e}")

print("\nTesting with valid statistics (should work):")
stats3 = [
    {"columns": [{"name": "a", "min": 1, "max": 5}]},
    {"columns": [{"name": "a", "min": 6, "max": 10}]},
]

try:
    result3 = sorted_columns(stats3)
    print(f"Result: {result3}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with None in second row group (should handle gracefully):")
stats4 = [
    {"columns": [{"name": "a", "min": 1, "max": 5}]},
    {"columns": [{"name": "a", "min": None, "max": None}]},
]

try:
    result4 = sorted_columns(stats4)
    print(f"Result: {result4}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with all None values (should skip):")
stats5 = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": None, "max": None}]},
]

try:
    result5 = sorted_columns(stats5)
    print(f"Result: {result5}")
except Exception as e:
    print(f"Error: {e}")