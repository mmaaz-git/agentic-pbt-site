#!/usr/bin/env python3
"""Simplified test to reproduce the pydantic_encoder bug with non-UTF-8 bytes"""

from pydantic.deprecated.json import pydantic_encoder
import warnings
import base64

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
    b'\x00',  # Null byte (valid UTF-8 but sometimes problematic)
    b'test\x80data',  # Mixed valid and invalid
]

for test_bytes in test_cases:
    try:
        result = pydantic_encoder(test_bytes)
        print(f"  {test_bytes!r}: Success - {result!r}")
    except UnicodeDecodeError as e:
        print(f"  {test_bytes!r}: UnicodeDecodeError - {str(e)[:50]}...")

# Test case 4: Binary data use cases
print("\nTest 4: Common binary data use cases")
# Simulate various binary data
binary_cases = [
    (b'\x89PNG\r\n\x1a\n', "PNG file header"),
    (b'\xff\xd8\xff', "JPEG file header"),
    (b'\x00\x01\x02\x03', "Binary protocol data"),
    (b'\x7f\x45\x4c\x46', "ELF file header"),
]

for data, description in binary_cases:
    try:
        result = pydantic_encoder(data)
        print(f"  {description}: Success - {result!r}")
    except UnicodeDecodeError:
        print(f"  {description}: Failed with UnicodeDecodeError")

# Test case 5: Show how bytes.decode() behaves
print("\nTest 5: Direct bytes.decode() behavior comparison")
test_bytes = b'\x80'
print(f"Testing with: {test_bytes!r}")

try:
    result = test_bytes.decode()
    print(f"  decode(): {result!r}")
except UnicodeDecodeError as e:
    print(f"  decode(): UnicodeDecodeError - {e}")

try:
    result = test_bytes.decode('utf-8', errors='replace')
    print(f"  decode('utf-8', errors='replace'): {result!r}")
except Exception as e:
    print(f"  decode with replace: {e}")

try:
    result = test_bytes.decode('utf-8', errors='ignore')
    print(f"  decode('utf-8', errors='ignore'): {result!r}")
except Exception as e:
    print(f"  decode with ignore: {e}")

# Show alternative encodings
encoded = base64.b64encode(test_bytes).decode('ascii')
print(f"  base64 encoding: {encoded}")
print(f"  hex encoding: {test_bytes.hex()}")

# Test case 6: Show that the current implementation is using decode()
print("\nTest 6: Verifying the implementation")
print("The encoder at line 55 of pydantic/deprecated/json.py uses:")
print("  bytes: lambda o: o.decode()")
print("This is equivalent to calling decode() with default UTF-8 and strict errors.")