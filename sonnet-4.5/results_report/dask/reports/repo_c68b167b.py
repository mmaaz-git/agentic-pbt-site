from dask.utils import key_split

# Test the failing case with non-UTF-8 bytes
result = key_split(b'\x80')
print(f"Result: {result}")