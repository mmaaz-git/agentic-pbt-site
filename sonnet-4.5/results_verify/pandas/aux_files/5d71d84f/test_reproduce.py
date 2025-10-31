import pandas as pd
import numpy as np

values1 = [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]
values2 = values1

s1 = pd.Series(values1)
s2 = pd.Series(values2)

result = s1.rolling(3).corr(s2)

print("Rolling correlation:", result.values)
print(f"Correlation at index 4: {result.iloc[4]}")
print(f"Is > 1.0? {result.iloc[4] > 1.0}")
print(f"Difference from 1.0: {result.iloc[4] - 1.0}")

# Also check if numpy gives the same result
print("\nFor comparison, numpy corrcoef for the window at index 4 (values [3.0, 1.0, 0.0]):")
window1 = [3.0, 1.0, 0.0]
window2 = [3.0, 1.0, 0.0]
numpy_corr = np.corrcoef(window1, window2)[0, 1]
print(f"NumPy correlation: {numpy_corr}")
print(f"NumPy correlation > 1.0? {numpy_corr > 1.0}")