import pandas as pd
import numpy as np

# Test with various large integers to understand the pattern
test_values = [
    (2**53, 1),      # Just at the boundary
    (2**53 + 1, 1),  # Just past the boundary
    (9007199254768175, 1),  # From the bug report
    (2**60, 1),      # Very large
]

for large_val, small_val in test_values:
    df = pd.DataFrame({
        'group': ['a', 'a'],
        'value': [large_val, small_val]
    })

    grouped_diff = df.groupby('group')['value'].diff()
    ungrouped_diff = df['value'].diff()

    print(f"\nTesting {large_val}, {small_val}:")
    print(f"  Large value loses precision in float64? {large_val != int(float(large_val))}")
    print(f"  Grouped diff:   {grouped_diff.loc[1]}")
    print(f"  Ungrouped diff: {ungrouped_diff.loc[1]}")
    print(f"  Match: {grouped_diff.loc[1] == ungrouped_diff.loc[1]}")

# Test what happens with multiple groups
print("\n--- Testing with multiple groups ---")
df2 = pd.DataFrame({
    'group': ['a', 'a', 'b', 'b'],
    'value': [9007199254768175, 1, 9007199254768175, 2]
})

grouped_diff2 = df2.groupby('group')['value'].diff()
ungrouped_diff2 = df2['value'].diff()

print("Grouped diff results:", grouped_diff2.tolist())
print("Ungrouped diff results:", ungrouped_diff2.tolist())

# Check if this is about float64 representation limits
print("\n--- Float64 precision limit ---")
print(f"2^53 = {2**53}")
print(f"2^53 as float64 = {float(2**53)}")
print(f"2^53 + 1 = {2**53 + 1}")
print(f"2^53 + 1 as float64 = {float(2**53 + 1)}")
print(f"Are they equal in float64? {float(2**53) == float(2**53 + 1)}")