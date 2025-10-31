import pandas as pd

df = pd.DataFrame({
    "group": [0] * 35,
    "value": [479349.0592509031] * 35
})

grouped = df.groupby("group")
min_result = grouped["value"].min()
max_result = grouped["value"].max()
mean_result = grouped["value"].mean()

print(f"Min:  {min_result[0]:.15f}")
print(f"Max:  {max_result[0]:.15f}")
print(f"Mean: {mean_result[0]:.15f}")
print(f"Mean > Max: {mean_result[0] > max_result[0]}")
print(f"Difference: {mean_result[0] - max_result[0]:.2e}")

# Additional verification
print(f"\nActual difference (precise): {mean_result[0] - max_result[0]}")
print(f"Min == Max: {min_result[0] == max_result[0]}")
print(f"All values identical: {all(df['value'] == df['value'][0])}")
print(f"Expected mean (same as values): {df['value'][0]:.15f}")