import pandas as pd
import numpy as np

# Test how pandas handles rolling/shifting empty data structures
print("Testing pandas behavior with empty structures:")

# Test 1: Empty Series
empty_series = pd.Series([])
print(f"\nEmpty Series: {empty_series}")
try:
    # Pandas uses shift for rolling-like operations
    shifted = empty_series.shift(1)
    print(f"Shifted by 1: {shifted}")
    print(f"Result is empty: {len(shifted) == 0}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Empty Index
empty_index = pd.Index([])
print(f"\nEmpty Index: {empty_index}")
# Pandas Index doesn't have a roll method, but we can simulate it with slicing
try:
    # Simulate rolling by concatenating slices
    shift = 1
    if len(empty_index) > 0:
        shift_amount = shift % len(empty_index)
        rolled = empty_index[-shift_amount:].append(empty_index[:-shift_amount])
    else:
        rolled = empty_index
    print(f"Simulated roll by 1: {rolled}")
    print(f"Result is empty: {len(rolled) == 0}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Empty DataFrame with rolling window
empty_df = pd.DataFrame()
print(f"\nEmpty DataFrame: {empty_df}")
try:
    rolled_df = empty_df.rolling(window=2).mean()
    print(f"Rolling window result: {rolled_df}")
    print(f"Result is empty: {len(rolled_df) == 0}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: NumPy roll on empty array
empty_array = np.array([])
print(f"\nEmpty numpy array: {empty_array}")
try:
    rolled_array = np.roll(empty_array, 1)
    print(f"np.roll result: {rolled_array}")
    print(f"Result is empty: {len(rolled_array) == 0}")
except Exception as e:
    print(f"Error: {e}")