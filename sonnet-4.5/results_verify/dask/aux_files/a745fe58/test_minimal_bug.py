import dask.base
import traceback

print("Testing minimal example: dask.base.key_split(b'\\x80')")
try:
    result = dask.base.key_split(b'\x80')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nTesting with valid UTF-8 bytes:")
try:
    result = dask.base.key_split(b'hello-world-1')
    print(f"Result for b'hello-world-1': {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with None:")
try:
    result = dask.base.key_split(None)
    print(f"Result for None: {result}")
except Exception as e:
    print(f"Error: {e}")