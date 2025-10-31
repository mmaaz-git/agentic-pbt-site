import pandas as pd
import xarray.indexes as xr_indexes

print("Creating empty pandas index...")
empty_pd_idx = pd.Index([])
print(f"Empty index created: {empty_pd_idx}")

print("\nCreating xarray PandasIndex...")
idx = xr_indexes.PandasIndex(empty_pd_idx, dim='x')
print(f"PandasIndex created: {idx}")

print("\nAttempting to roll by 1...")
try:
    result = idx.roll({'x': 1})
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()