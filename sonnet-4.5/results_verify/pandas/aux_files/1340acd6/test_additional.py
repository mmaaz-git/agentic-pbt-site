import pandas as pd
import numpy as np

# Test various ranges to find the boundary
print("Testing various small ranges to find the boundary:")
print("-" * 50)

test_cases = [
    (1e-300, "1e-300"),
    (1e-305, "1e-305"),
    (1e-308, "1e-308"),
    (1e-310, "1e-310"),
    (1e-313, "1e-313"),
    (2.225e-313, "2.225e-313"),
    (1e-315, "1e-315"),
]

for val, label in test_cases:
    values = [0.0, val]
    s = pd.Series(values)
    try:
        result = pd.cut(s, bins=2)
        has_nan = result.isna().any()
        all_nan = result.isna().all()
        print(f"Range {label:15} -> NaN: {'All' if all_nan else ('Some' if has_nan else 'None')}")
        if not all_nan and not has_nan:
            print(f"    Result: {result.tolist()}")
    except Exception as e:
        print(f"Range {label:15} -> Error: {type(e).__name__}")

print("\nTesting with normal float range for comparison:")
print("-" * 50)
values = [0.0, 1.0]
s = pd.Series(values)
result = pd.cut(s, bins=2)
print(f"Input: {values}")
print(f"Result: {result.tolist()}")
print(f"Categories: {result.cat.categories.tolist()}")