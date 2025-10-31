from dask.dataframe.io.io import sorted_division_locations

# Example directly from the docstring
L = ['A', 'B', 'C', 'D', 'E', 'F']
try:
    divisions, locations = sorted_division_locations(L, chunksize=2)
    print(f"Success: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")