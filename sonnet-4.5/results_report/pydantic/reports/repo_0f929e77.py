from pydantic.deprecated.json import ENCODERS_BY_TYPE

# Test case: non-UTF-8 bytes
non_utf8_bytes = b'\x80'
encoder = ENCODERS_BY_TYPE[bytes]

try:
    result = encoder(non_utf8_bytes)
    print(f"Success: {result!r}")
except Exception as e:
    print(f"{e.__class__.__name__}: {e}")