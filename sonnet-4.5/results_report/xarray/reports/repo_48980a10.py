from xarray.indexes import RangeIndex

# This should demonstrate the ZeroDivisionError
index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
print(f"Index created: {index}")