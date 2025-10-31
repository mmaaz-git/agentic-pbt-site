from dask.utils import key_split

# Test the specific failing case
invalid_utf8 = b'\x80'
try:
    result = key_split(invalid_utf8)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")

# Test with valid UTF-8 bytes (as shown in documentation)
valid_utf8 = b'hello-world-1'
try:
    result = key_split(valid_utf8)
    print(f"Valid UTF-8 result: {result}")
except Exception as e:
    print(f"Error with valid UTF-8: {e}")

# Test with None to see what 'Other' behavior is
result_none = key_split(None)
print(f"Result for None: {result_none}")