import pandas as pd

# Test case from bug report
df = pd.DataFrame({
    'group': ['a', 'a'],
    'value': [9007199254768175, 1]
})

grouped_diff = df.groupby('group')['value'].diff()
ungrouped_diff = df['value'].diff()

print(f"Original values: {df['value'].tolist()}")
print(f"Grouped diff():   {grouped_diff.loc[1]}")
print(f"Ungrouped diff(): {ungrouped_diff.loc[1]}")
print(f"Match: {grouped_diff.loc[1] == ungrouped_diff.loc[1]}")
print(f"Difference: {abs(grouped_diff.loc[1] - ungrouped_diff.loc[1])}")

# Let's also check what happens when we convert to float first
print("\n--- Additional checks ---")
print(f"Large value as int64: {df['value'].iloc[0]}")
print(f"Large value as float64: {float(df['value'].iloc[0])}")
print(f"Precision loss? {df['value'].iloc[0] != int(float(df['value'].iloc[0]))}")

# Manual calculation
manual_diff_int = df['value'].iloc[1] - df['value'].iloc[0]
manual_diff_float_first = float(df['value'].iloc[1]) - float(df['value'].iloc[0])
print(f"\nManual diff (int first): {manual_diff_int}")
print(f"Manual diff (float first): {manual_diff_float_first}")
print(f"Manual diff (int first) as float: {float(manual_diff_int)}")