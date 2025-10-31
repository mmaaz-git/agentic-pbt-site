#!/usr/bin/env python3
"""Test how other dask.bag statistical functions handle empty bags"""

import dask
import dask.bag as db
import math

dask.config.set(scheduler='synchronous')

print("Testing dask.bag statistical functions with empty bags:")
print("-" * 50)

# Create an empty bag
empty_bag = db.from_sequence([], npartitions=1)

# Test various statistical functions
functions_to_test = [
    ('count', lambda b: b.count()),
    ('mean', lambda b: b.mean()),
    ('std', lambda b: b.std()),
    ('var', lambda b: b.var()),
    ('sum', lambda b: b.sum()),
    ('min', lambda b: b.min()),
    ('max', lambda b: b.max()),
]

for func_name, func in functions_to_test:
    print(f"\n{func_name}():")
    try:
        result = func(empty_bag).compute()
        print(f"  Result: {result}")
        if isinstance(result, float) and math.isnan(result):
            print(f"  (NaN)")
    except ZeroDivisionError as e:
        print(f"  ZeroDivisionError: {e}")
    except ValueError as e:
        print(f"  ValueError: {e}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

# Also test with a non-empty bag for comparison
print("\n" + "-" * 50)
print("For comparison, with non-empty bag [1, 2, 3]:")
print("-" * 50)
non_empty_bag = db.from_sequence([1, 2, 3], npartitions=1)

for func_name, func in functions_to_test:
    try:
        result = func(non_empty_bag).compute()
        print(f"{func_name}(): {result}")
    except Exception as e:
        print(f"{func_name}(): {type(e).__name__}: {e}")