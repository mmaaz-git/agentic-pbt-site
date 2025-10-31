import pandas as pd
import time

# Create a test DataFrame with mixed types
df = pd.DataFrame({
    'int_col': list(range(100000)),
    'float_col': [float(i) * 0.1 for i in range(100000)],
    'str_col': [f'value_{i}' for i in range(100000)]
})

print("Testing pandas.DataFrame.to_dict() with 'split' and 'tight' orientations")
print("=" * 70)
print(f"DataFrame shape: {df.shape}")
print(f"Column types: {df.dtypes.to_dict()}")
print()

# Test 'split' orientation
start = time.time()
split_result = df.to_dict(orient='split')
split_time = time.time() - start
print(f"'split' orientation time: {split_time:.4f} seconds")

# Test 'tight' orientation
start = time.time()
tight_result = df.to_dict(orient='tight')
tight_time = time.time() - start
print(f"'tight' orientation time: {tight_time:.4f} seconds")

print()
print(f"Performance difference: 'tight' is {tight_time/split_time:.1f}x slower than 'split'")

# Verify that the data is the same
print()
print("Data comparison:")
print(f"Data values identical: {split_result['data'] == tight_result['data']}")

# Show the structural difference between split and tight
print()
print("Keys in 'split' result:", list(split_result.keys()))
print("Keys in 'tight' result:", list(tight_result.keys()))

# Demonstrate that 'tight' has additional metadata
if 'index_names' in tight_result:
    print(f"'tight' includes index_names: {tight_result['index_names']}")
if 'column_names' in tight_result:
    print(f"'tight' includes column_names: {tight_result['column_names']}")