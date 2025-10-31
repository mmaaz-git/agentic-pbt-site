import pandas as pd
import numpy as np

data = [1.023075029544998, 524288.3368640885, 0.0]
s = pd.Series(data)
result = s.expanding().sum()

print(f"Input data: {data}")
print(f"Expanding sum results:")
for i in range(len(result)):
    print(f"  Position {i}: {result.iloc[i]:.20f}")

print(f"\nComparison:")
print(f"Position 1 sum: {result.iloc[1]:.20f}")
print(f"Position 2 sum: {result.iloc[2]:.20f}")
print(f"Difference: {result.iloc[1] - result.iloc[2]:.20e}")

# Check monotonicity
print(f"\nMonotonicity check:")
print(f"result[2] >= result[1]: {result.iloc[2] >= result.iloc[1]}")

# Compare with numpy cumsum
numpy_result = np.cumsum(data)
print(f"\nNumPy cumsum results:")
for i in range(len(numpy_result)):
    print(f"  Position {i}: {numpy_result[i]:.20f}")

# Compare with Python sum
print(f"\nPython sum results:")
for i in range(len(data)):
    python_sum = sum(data[:i+1])
    print(f"  Position {i}: {python_sum:.20f}")

assert result.iloc[2] >= result.iloc[1], "Monotonicity violated!"