import pandas as pd

# Test 1: Simple reproduction
print("Test 1: Simple reproduction")
df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

percentiles = [0.01, 0.010000000000000002]
print(f"Percentiles: {percentiles}")
print(f"Are they different? {percentiles[0] != percentiles[1]}")

try:
    result = df.describe(percentiles=percentiles)
    print("Success - no error occurred")
    print(result)
except ValueError as e:
    print(f"ValueError: {e}")

# Test 2: Check format_percentiles directly
print("\nTest 2: Check format_percentiles directly")
from pandas.core.methods.describe import format_percentiles

formatted = format_percentiles(percentiles)
print(f"Formatted percentiles: {formatted}")
print(f"Are formatted labels unique? {len(formatted) == len(set(formatted))}")