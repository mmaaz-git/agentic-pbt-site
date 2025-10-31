import pandas as pd
import dask.dataframe as dd

# Check pandas behavior with different dtype_backend settings
s = pd.Series(['0', '1', '2'])

print("Testing pandas with different dtype_backend settings:")

# Default pandas behavior (no dtype_backend)
result_default = pd.to_numeric(s)
print(f"pandas default (no dtype_backend): {result_default.dtype}")

# With numpy_nullable backend
try:
    result_nullable = pd.to_numeric(s, dtype_backend='numpy_nullable')
    print(f"pandas with dtype_backend='numpy_nullable': {result_nullable.dtype}")
except Exception as e:
    print(f"Error with numpy_nullable: {e}")

# Without any backend explicitly set to None
try:
    import pandas._libs.lib
    result_no_backend = pd.to_numeric(s, dtype_backend=pandas._libs.lib.no_default)
    print(f"pandas with dtype_backend=no_default: {result_no_backend.dtype}")
except Exception as e:
    print(f"Error with no_default: {e}")

# Check what dask is doing
print("\n\nChecking dask behavior:")
ds = dd.from_pandas(s, npartitions=2)
result_dask = dd.to_numeric(ds).compute()
print(f"dask result: {result_dask.dtype}")

# Check if dask accepts dtype_backend
print("\nChecking if dask accepts dtype_backend parameter:")
import inspect
sig = inspect.signature(dd.to_numeric)
print(f"Dask to_numeric signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")