import pandas as pd
from xarray.core.indexes import PandasIndex

# Create an empty pandas index
empty_idx = pd.Index([])

# Create a PandasIndex with the empty index
xr_idx = PandasIndex(empty_idx, "x")

print(f"Index length: {len(xr_idx.index)}")
print(f"Index shape: {xr_idx.index.shape}")
print(f"Attempting to roll by 1...")

# Attempt to roll the empty index - this will crash with ZeroDivisionError
rolled = xr_idx.roll({"x": 1})
print("Roll succeeded!")