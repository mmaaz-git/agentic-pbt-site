#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.formats.format import format_percentiles

# Test case from bug report
print("Test case 1: format_percentiles([0.99999, 0.0])")
result = format_percentiles([0.99999, 0.0])
print(f"Result: {result}")
print(f"0.99999 formatted as: {result[0]}")
print(f"0.0 formatted as: {result[1]}")
print()

# Check if it violates the contract
if result[0] == "100%":
    print("VIOLATION: 0.99999 was rounded to 100%")
else:
    print("OK: 0.99999 was not rounded to 100%")

print()

# Test the example from documentation
print("Test case 2: Documentation example")
result2 = format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
print(f"Input: [0.01999, 0.02001, 0.5, 0.666666, 0.9999]")
print(f"Result: {result2}")
print(f"Expected: ['1.999%', '2.001%', '50%', '66.667%', '99.99%']")
print()

# Test more edge cases
print("Test case 3: Values close to 0")
result3 = format_percentiles([0.00001, 0.0001, 0.001])
print(f"Input: [0.00001, 0.0001, 0.001]")
print(f"Result: {result3}")
print()

print("Test case 4: Values close to 1")
result4 = format_percentiles([0.999, 0.9999, 0.99999, 0.999999])
print(f"Input: [0.999, 0.9999, 0.99999, 0.999999]")
print(f"Result: {result4}")
print()

# Test exact 0 and 1
print("Test case 5: Exact 0 and 1 (should be allowed)")
result5 = format_percentiles([0.0, 0.5, 1.0])
print(f"Input: [0.0, 0.5, 1.0]")
print(f"Result: {result5}")
print()

# Additional test cases to understand the rounding behavior
print("Test case 6: Understanding the int_idx logic")
import numpy as np
test_vals = [0.99999, 0.5, 0.25]
percentiles = np.asarray(test_vals)
percentiles = 100 * percentiles

# Simulating the function's logic
from pandas.io.formats.format import get_precision
prec = get_precision(percentiles)
percentiles_round_type = percentiles.round(prec).astype(int)
int_idx = np.isclose(percentiles_round_type, percentiles)

print(f"Input: {test_vals}")
print(f"After *100: {percentiles}")
print(f"Precision: {prec}")
print(f"Round type: {percentiles_round_type}")
print(f"int_idx (which ones are close to integers): {int_idx}")
print(f"Values considered 'close to integer': {percentiles[int_idx]}")