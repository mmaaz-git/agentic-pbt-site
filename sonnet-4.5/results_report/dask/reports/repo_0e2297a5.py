import dask.base

# Test case that crashes with UnicodeDecodeError
result = dask.base.key_split(b'\x80')
print(f"Result: {result}")