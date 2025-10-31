#!/usr/bin/env python3
from pandas._config import get_option

print(f"display.encoding: {get_option('display.encoding')}")

# Also test with different invalid bytes
from pandas.core.computation.common import ensure_decoded

test_cases = [
    b'\x80',  # Invalid start byte
    b'\xc0\x80',  # Overlong encoding
    b'\xed\xa0\x80',  # Surrogate half
    b'Hello\x80World',  # Valid UTF-8 with invalid byte in middle
]

for test_bytes in test_cases:
    print(f"\nTesting: {test_bytes!r}")
    try:
        result = ensure_decoded(test_bytes)
        print(f"  Success: {result!r}")
    except UnicodeDecodeError as e:
        print(f"  UnicodeDecodeError: {e}")