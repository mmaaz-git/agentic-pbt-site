from xarray.indexes import RangeIndex

# Test case that crashes with num=1 and endpoint=True
index = RangeIndex.linspace(
    start=0.0,
    stop=1.0,
    num=1,
    endpoint=True,
    dim="x"
)
print(f"Created index with size {index.size}")