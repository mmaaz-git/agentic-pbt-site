import pandas as pd
import numpy as np

values = [1.1125369292536007e-308, -6.312184867281418e-301]
x = pd.Series(values)

print(f"Input values: {x.tolist()}")
print(f"All values are valid (non-NaN): {x.notna().all()}")
print(f"Data range: [{x.min()}, {x.max()}]")
print()

try:
    result = pd.cut(x, bins=2)
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")