from dask.utils import key_split
import sys

# Test various inputs to understand the function's behavior

# Test cases from documentation
test_cases = [
    ('x', 'x'),
    ('x-1', 'x'),
    ('x-1-2-3', 'x'),
    (('x-2', 1), 'x'),
    ("('x-2', 1)", 'x'),
    ("('x', 1)", 'x'),
    ('hello-world-1', 'hello-world'),
    (b'hello-world-1', 'hello-world'),  # bytes example from docs
    ('ae05086432ca935f6eba409a8ecd4896', 'data'),
    ('<module.submodule.myclass object at 0xdaf372', 'myclass'),
    (None, 'Other'),
    ('x-abcdefab', 'x'),
    ('_(x)', 'x'),
]

print("Testing documented examples:")
for input_val, expected in test_cases:
    try:
        result = key_split(input_val)
        status = "✓" if result == expected else "✗"
        print(f"{status} key_split({repr(input_val)}) = {repr(result)} (expected: {repr(expected)})")
    except Exception as e:
        print(f"✗ key_split({repr(input_val)}) raised {type(e).__name__}: {e}")

# Test edge cases with bytes
print("\nTesting various byte inputs:")
byte_tests = [
    (b'hello', 'Valid ASCII bytes'),
    (b'hello-123', 'Valid ASCII with hyphen'),
    (b'\xc3\xa9', 'Valid UTF-8 (é)'),
    (b'\x80', 'Invalid UTF-8 (single byte)'),
    (b'\xc3\x28', 'Invalid UTF-8 sequence'),
    (b'\xff\xfe', 'Invalid UTF-8 (BOM-like)'),
]

for byte_input, description in byte_tests:
    try:
        result = key_split(byte_input)
        print(f"✓ key_split({repr(byte_input)}) = {repr(result)} - {description}")
    except UnicodeDecodeError as e:
        print(f"✗ key_split({repr(byte_input)}) raised UnicodeDecodeError - {description}")
    except Exception as e:
        print(f"✗ key_split({repr(byte_input)}) raised {type(e).__name__}: {e} - {description}")

# Test what happens with other problematic inputs
print("\nTesting other edge cases:")
edge_cases = [
    (123, "Integer"),
    ([1, 2, 3], "List"),
    ({'key': 'value'}, "Dict"),
    (object(), "Generic object"),
]

for input_val, description in edge_cases:
    try:
        result = key_split(input_val)
        print(f"key_split({description}) = {repr(result)}")
    except Exception as e:
        print(f"key_split({description}) raised {type(e).__name__}: {e}")