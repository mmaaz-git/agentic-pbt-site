import pandas as pd
import xarray.indexes as xr_indexes

# Create an empty pandas index
empty_pd_idx = pd.Index([])

# Create a PandasIndex object with the empty index
idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')

# Try to roll the empty index by 1 position
try:
    result = idx.roll({'x': 1})
    print("Roll succeeded. Result:", result)
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()