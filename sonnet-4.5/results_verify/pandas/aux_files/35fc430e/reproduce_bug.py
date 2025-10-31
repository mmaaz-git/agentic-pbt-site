from pandas.api.indexers import FixedForwardWindowIndexer
import pandas as pd

print("=" * 50)
print("Test 1: Basic reproduction with window_size=-1")
print("=" * 50)

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"start[0]={start[0]}, end[0]={end[0]} -> valid: {start[0] <= end[0]}")
print(f"start[1]={start[1]}, end[1]={end[1]} -> valid: {start[1] <= end[1]}")

# Check the assertion
try:
    assert start[1] > end[1]
    print("✓ Confirmed: start[1] > end[1] (invalid window bounds)")
except AssertionError:
    print("✗ The bug was not reproduced")

print("\n" + "=" * 50)
print("Test 2: Effect on DataFrame rolling operations")
print("=" * 50)

df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
print("Original DataFrame:")
print(df)

indexer = FixedForwardWindowIndexer(window_size=-1)
result = df.rolling(window=indexer).sum()
print("\nResult of rolling sum with window_size=-1:")
print(result)

# Verify all zeros
all_zeros = all(result['A'] == 0.0)
print(f"\nAll values are zero: {all_zeros}")

print("\n" + "=" * 50)
print("Test 3: Compare with positive window_size=2")
print("=" * 50)

indexer_positive = FixedForwardWindowIndexer(window_size=2)
result_positive = df.rolling(window=indexer_positive).sum()
print("Result of rolling sum with window_size=2:")
print(result_positive)

print("\n" + "=" * 50)
print("Test 4: More negative window sizes")
print("=" * 50)

for window_size in [-1, -2, -5]:
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=5)
    print(f"\nwindow_size={window_size}:")
    print(f"  start: {start}")
    print(f"  end:   {end}")

    # Check for invalid bounds
    invalid_indices = []
    for i in range(len(start)):
        if start[i] > end[i]:
            invalid_indices.append(i)

    if invalid_indices:
        print(f"  Invalid bounds at indices: {invalid_indices}")