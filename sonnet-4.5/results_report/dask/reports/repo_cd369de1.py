from dask.utils import key_split

# Test with non-UTF-8 bytes
try:
    result = key_split(b'\x80')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with valid UTF-8 bytes for comparison
try:
    result = key_split(b'hello-world-1')
    print(f"Valid UTF-8 result: {result}")
except Exception as e:
    print(f"Error with valid UTF-8: {type(e).__name__}: {e}")