#!/usr/bin/env python3
"""Test to reproduce the pydantic_encoder bug with non-UTF-8 bytes"""

from hypothesis import given, strategies as st
from pydantic.deprecated.json import pydantic_encoder
import warnings

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("=" * 50)
print("Testing pydantic_encoder with non-UTF-8 bytes")
print("=" * 50)

# Test case 1: The specific failing input from bug report
print("\nTest 1: Specific failing input b'\\x80'")
invalid_utf8_bytes = b'\x80'
try:
    result = pydantic_encoder(invalid_utf8_bytes)
    print(f"Success: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")

# Test case 2: Valid UTF-8 bytes
print("\nTest 2: Valid UTF-8 bytes b'hello'")
valid_utf8_bytes = b'hello'
try:
    result = pydantic_encoder(valid_utf8_bytes)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")

# Test case 3: More examples of invalid UTF-8
print("\nTest 3: More invalid UTF-8 sequences")
test_cases = [
    b'\xff',  # Invalid start byte
    b'\x80\x80',  # Invalid continuation
    b'\xc0\x80',  # Overlong encoding
    b'\xed\xa0\x80',  # UTF-16 surrogate
]

for test_bytes in test_cases:
    try:
        result = pydantic_encoder(test_bytes)
        print(f"  {test_bytes!r}: Success - {result}")
    except UnicodeDecodeError as e:
        print(f"  {test_bytes!r}: UnicodeDecodeError")

# Test case 4: Run the hypothesis test
print("\nTest 4: Running hypothesis test")
failures = []

@given(st.binary(min_size=1, max_size=10))
def test_pydantic_encoder_bytes_decode(b):
    try:
        result = pydantic_encoder(b)
        assert isinstance(result, str)
        assert result == b.decode()
    except UnicodeDecodeError:
        failures.append(b)

# Run a limited number of examples
from hypothesis import settings
test_pydantic_encoder_bytes_decode_limited = test_pydantic_encoder_bytes_decode.hypothesis.with_settings(max_examples=100)
test_pydantic_encoder_bytes_decode_limited()

if failures:
    print(f"Found {len(failures)} failing cases out of 100 tested")
    print(f"First 5 failures: {failures[:5]}")
else:
    print("No failures found in 100 test cases")

# Test case 5: Check what bytes.decode() does with these inputs
print("\nTest 5: Direct bytes.decode() behavior")
test_bytes = b'\x80'
try:
    result = test_bytes.decode()
    print(f"bytes.decode() succeeded: {result}")
except UnicodeDecodeError as e:
    print(f"bytes.decode() failed: {e}")

try:
    result = test_bytes.decode('utf-8', errors='replace')
    print(f"bytes.decode('utf-8', errors='replace') succeeded: {result!r}")
except Exception as e:
    print(f"bytes.decode with replace failed: {e}")

import base64
encoded = base64.b64encode(test_bytes).decode('ascii')
print(f"base64 encoding would give: {encoded}")