import pandas as pd
import numpy as np

# Minimal test case that demonstrates negative variance
data = [3222872787.0, 0.0, 2.0, 0.0]
s = pd.Series(data)
var = s.rolling(window=3).var()

print("Input data:", data)
print("\nRolling variance (window=3):")
print(var)
print("\nDetailed analysis:")
for i in range(len(var)):
    if pd.notna(var.iloc[i]):
        if i >= 2:
            window_data = data[i-2:i+1]
            print(f"Index {i}: window={window_data}, variance={var.iloc[i]:.6f}")

            # Manual calculation for verification
            mean = sum(window_data) / len(window_data)
            manual_var = sum((x - mean)**2 for x in window_data) / (len(window_data) - 1)
            print(f"  Manual calculation: mean={mean:.6f}, variance={manual_var:.6f}")

            # NumPy calculation for comparison
            numpy_var = np.var(window_data, ddof=1)
            print(f"  NumPy calculation: variance={numpy_var:.6f}")

            if var.iloc[i] < 0:
                print(f"  ❌ NEGATIVE VARIANCE DETECTED: {var.iloc[i]:.6f}")