from pydantic.v1.types import ByteSize

# Test direct instantiation
print("Direct instantiation:")
print(f"ByteSize(0.5) = {ByteSize(0.5)}")
print(f"ByteSize(1.7) = {ByteSize(1.7)}")
print(f"ByteSize(2.5) = {ByteSize(2.5)}")
print()

# Test validation with fractional strings
print("Validation with fractional strings:")
print(f"ByteSize.validate('0.5b') = {ByteSize.validate('0.5b')}")
print(f"ByteSize.validate('1.7kb') = {ByteSize.validate('1.7kb')}")
print(f"ByteSize.validate('2.5mb') = {ByteSize.validate('2.5mb')}")
print()

# Test what the regex accepts
import re
byte_string_re = re.compile(r'^\s*(\d*\.?\d+)\s*(\w+)?', re.IGNORECASE)
test_strings = ["0.5b", "1.7kb", "2.5mb", "0.1b", "0.9999b"]
print("Regex matches:")
for s in test_strings:
    match = byte_string_re.match(s)
    if match:
        scalar, unit = match.groups()
        print(f"  '{s}' -> scalar='{scalar}', unit='{unit}'")