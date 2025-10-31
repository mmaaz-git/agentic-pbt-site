import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume

# Test with the specific failing input
print("Testing with the failing input...")
values1 = [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0]
values2 = values1

s1 = pd.Series(values1)
s2 = pd.Series(values2)
result = s1.rolling(3).corr(s2)

for i, val in enumerate(result):
    if not np.isnan(val):
        try:
            assert -1 <= val <= 1, f"At index {i}: correlation {val} outside [-1, 1]"
        except AssertionError as e:
            print(f"Test failed: {e}")
            break
else:
    print("Test passed")