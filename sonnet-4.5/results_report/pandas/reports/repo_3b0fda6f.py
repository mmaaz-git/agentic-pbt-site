from pandas.io.parsers.readers import _validate_names
import math

# Test with duplicate NaN values
names = [float('nan'), float('nan')]

print("Testing _validate_names with duplicate NaN values:")
print(f"Input: {names}")
print(f"len(names): {len(names)}")
print(f"len(set(names)): {len(set(names))}")
print(f"set(names): {set(names)}")

# Check if NaN != NaN
print(f"\nNaN equality check: nan == nan -> {float('nan') == float('nan')}")
print(f"NaN identity check: nan is nan -> {float('nan') is float('nan')}")

# Try to validate the names
print("\nCalling _validate_names(names)...")
try:
    _validate_names(names)
    print("Result: ACCEPTED (no exception raised) - BUG CONFIRMED")
except ValueError as e:
    print(f"Result: REJECTED with error: {e}")

# Test with regular duplicate values for comparison
print("\n--- Testing with regular duplicates for comparison ---")
regular_duplicates = ['col1', 'col1']
print(f"Input: {regular_duplicates}")
print("Calling _validate_names(regular_duplicates)...")
try:
    _validate_names(regular_duplicates)
    print("Result: ACCEPTED (no exception raised)")
except ValueError as e:
    print(f"Result: REJECTED with error: {e}")

# Test with mixed NaN and regular values
print("\n--- Testing with mixed NaN and regular values ---")
mixed_with_nan = [float('nan'), 'col1', float('nan'), 'col2']
print(f"Input: {mixed_with_nan}")
print("Calling _validate_names(mixed_with_nan)...")
try:
    _validate_names(mixed_with_nan)
    print("Result: ACCEPTED (no exception raised)")
except ValueError as e:
    print(f"Result: REJECTED with error: {e}")

# Real-world impact: Creating a DataFrame with duplicate NaN column names
print("\n--- Real-world impact test ---")
import pandas as pd
from io import StringIO

csv = StringIO("1,2\n3,4")
print("Creating DataFrame with pd.read_csv using duplicate NaN column names...")
try:
    df = pd.read_csv(csv, names=[float('nan'), float('nan')])
    print("Result: DataFrame created successfully - BUG IMPACT CONFIRMED")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame:\n{df}")
except ValueError as e:
    print(f"Result: Failed with error: {e}")