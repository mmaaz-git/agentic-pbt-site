from pydantic.deprecated.json import ENCODERS_BY_TYPE, pydantic_encoder

non_utf8_bytes = b'\x80'
encoder = ENCODERS_BY_TYPE[bytes]

print(f"Testing with non-UTF-8 bytes: {non_utf8_bytes!r}")
try:
    result = encoder(non_utf8_bytes)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    print("This matches the bug report - UnicodeDecodeError when encoding non-UTF-8 bytes")

# Test with valid UTF-8 bytes
valid_utf8_bytes = b'Hello, World!'
print(f"\nTesting with valid UTF-8 bytes: {valid_utf8_bytes!r}")
try:
    result = encoder(valid_utf8_bytes)
    print(f"Result: {result!r}")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"Error: {e}")