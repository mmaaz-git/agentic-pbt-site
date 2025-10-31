from xarray.indexes.range_index import RangeIndex

# This should work like numpy.linspace(0.0, 1.0, num=1, endpoint=True)
# which returns array([0.])
index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")