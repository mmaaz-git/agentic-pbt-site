import numpy as np

print("Testing NumPy variance with different ddof values:")
print("=" * 50)

data = np.array([1.0, 2.0])
print(f"Data: {data}")
print(f"Data size (n): {len(data)}")
print()

for ddof in range(-2, 5):
    try:
        result = np.var(data, ddof=ddof)
        print(f"ddof={ddof}: variance = {result}")
        if result < 0:
            print(f"  WARNING: Negative variance!")
    except Exception as e:
        print(f"ddof={ddof}: ERROR - {type(e).__name__}: {e}")

print("\nComparing with pandas:")
import pandas as pd

s = pd.Series([1.0, 2.0])
print(f"pandas Series: {list(s)}")
for ddof in range(-2, 5):
    try:
        result = s.var(ddof=ddof)
        print(f"ddof={ddof}: variance = {result}")
        if result < 0:
            print(f"  WARNING: Negative variance!")
    except Exception as e:
        print(f"ddof={ddof}: ERROR - {type(e).__name__}: {e}")