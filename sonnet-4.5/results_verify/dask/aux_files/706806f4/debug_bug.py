#!/usr/bin/env python3
"""Debug the exact location of the bug"""

from dask.dataframe.io.parquet.core import sorted_columns
import traceback

statistics = [{"columns": [{"name": "a", "min": 0, "max": None}]}]

print("Debugging the bug...")
print("Input statistics:", statistics)
print()

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print("Exception caught!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing what happens when we try to sort a list with None")
print("=" * 60)

test_lists = [
    [0, None],
    [None, 0],
    [0, 5, None],
    [0, None, 10],
]

for test_list in test_lists:
    print(f"\nTrying to sort: {test_list}")
    try:
        sorted_list = sorted(test_list)
        print(f"Sorted result: {sorted_list}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing comparison of None with integers")
print("=" * 60)

comparisons = [
    ("None < 0", None < 0),
    ("None > 0", None > 0),
    ("None == 0", None == 0),
    ("0 < None", 0 < None),
]

for description, comparison in comparisons:
    print(f"\nTrying: {description}")
    try:
        result = comparison
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")