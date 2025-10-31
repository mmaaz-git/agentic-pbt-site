from dask.utils import key_split

invalid_utf8_bytes = b'\x80'
result = key_split(invalid_utf8_bytes)
print(f"Result: {result}")