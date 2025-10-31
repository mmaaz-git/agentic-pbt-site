from dask.utils import key_split

# Test with invalid UTF-8 byte sequence
invalid_utf8 = b'\x80'
print(f"Testing key_split with invalid UTF-8 bytes: {invalid_utf8!r}")
try:
    result = key_split(invalid_utf8)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Additional test with another invalid UTF-8 sequence
invalid_utf8_2 = b'\xc3\x28'
print(f"Testing key_split with another invalid UTF-8 sequence: {invalid_utf8_2!r}")
try:
    result = key_split(invalid_utf8_2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with valid UTF-8 bytes for comparison
valid_utf8 = b'hello-world-1'
print(f"Testing key_split with valid UTF-8 bytes: {valid_utf8!r}")
try:
    result = key_split(valid_utf8)
    print(f"Result: {result!r}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")