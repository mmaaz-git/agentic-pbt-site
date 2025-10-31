#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.formats.format import format_percentiles
import numpy as np

# Test different edge cases
test_cases = [
    ([0.9995], "Testing 0.9995"),
    ([0.99995], "Testing 0.99995"),
    ([0.999995], "Testing 0.999995"),
    ([0.0005], "Testing 0.0005"),
    ([0.00005], "Testing 0.00005"),
    ([0.000005], "Testing 0.000005"),
    ([0.995, 0.9995], "Testing when values differ slightly"),
    ([0.99999, 0.99998], "Testing very close values near 100"),
]

for values, description in test_cases:
    print(f"\n{description}")
    print(f"Input: {values}")
    result = format_percentiles(values)
    print(f"Output: {result}")

    # Check for violations
    for val, res in zip(values, result):
        if val != 0.0 and res == "0%":
            print(f"  VIOLATION: {val} rounded to 0%")
        if val != 1.0 and res == "100%":
            print(f"  VIOLATION: {val} rounded to 100%")

# Test the specific failing case from the bug report more thoroughly
print("\n\nDetailed analysis of [0.99999, 0.0]:")
vals = [0.99999, 0.0]
percentiles = np.asarray(vals)
print(f"Original input: {vals}")

percentiles = 100 * percentiles
print(f"After *100: {percentiles}")

from pandas.io.formats.format import get_precision

# First precision calculation
prec = get_precision(percentiles)
print(f"First precision: {prec}")

percentiles_round_type = percentiles.round(prec).astype(int)
print(f"Round type: {percentiles_round_type}")

int_idx = np.isclose(percentiles_round_type, percentiles)
print(f"int_idx: {int_idx}")

# This is the problematic line 1603
out = np.empty_like(percentiles, dtype=object)
out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)

print(f"\nLine 1603 effect:")
print(f"  percentiles[int_idx]: {percentiles[int_idx]}")
print(f"  .round(): {percentiles[int_idx].round()}")
print(f"  .astype(int): {percentiles[int_idx].round().astype(int)}")
print(f"  Final: {out[int_idx]}")

print(f"\nThis is where 99.999 becomes '100' incorrectly!")