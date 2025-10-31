#!/usr/bin/env python3
"""Test script to reproduce the reported bug in dask.utils.key_split"""

from hypothesis import given, settings, strategies as st
from dask.utils import key_split

# Test the simple examples from the bug report
print("Testing simple examples from bug report:")
print(f"key_split('task-abcdefab') = '{key_split('task-abcdefab')}'")
print(f"key_split('task-12345678') = '{key_split('task-12345678')}'")

# Test the assertions
print("\nTesting assertions:")
try:
    assert key_split('task-abcdefab') == 'task'
    print("✓ key_split('task-abcdefab') == 'task' (PASSED)")
except AssertionError as e:
    print(f"✗ key_split('task-abcdefab') == 'task' (FAILED): {e}")

try:
    assert key_split('task-12345678') == 'task'
    print("✓ key_split('task-12345678') == 'task' (PASSED)")
except AssertionError as e:
    print(f"✗ key_split('task-12345678') == 'task' (FAILED)")

# Test more hex patterns
print("\nTesting various 8-character hex patterns:")
test_cases = [
    ('x-abcdefab', 'x'),  # All letters (a-f)
    ('x-12345678', 'x'),  # All digits
    ('x-1234abcd', 'x'),  # Mixed digits and letters
    ('x-abcd1234', 'x'),  # Mixed letters and digits
    ('x-00000000', 'x'),  # All zeros
    ('x-ffffffff', 'x'),  # All f's
    ('x-deadbeef', 'x'),  # Classic hex pattern
]

for key, expected in test_cases:
    result = key_split(key)
    status = "✓" if result == expected else "✗"
    print(f"{status} key_split('{key}') = '{result}', expected '{expected}'")

# Test the property-based test
print("\nRunning property-based test with limited examples:")

@given(
    key_base=st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    hex_suffix=st.text(min_size=8, max_size=8, alphabet='0123456789abcdef'),
)
@settings(max_examples=50)  # Limited for quick testing
def test_key_split_removes_8char_hex_suffix(key_base, hex_suffix):
    key = f"{key_base}-{hex_suffix}"
    result = key_split(key)
    expected = key_base
    assert result == expected, f"key_split('{key}') = '{result}', expected '{expected}' (hex suffix '{hex_suffix}' should be stripped)"

try:
    test_key_split_removes_8char_hex_suffix()
    print("Property-based test completed successfully")
except AssertionError as e:
    print(f"Property-based test failed: {e}")

# Let's also check what the docstring examples actually do
print("\nChecking docstring examples:")
print(f"key_split('x-abcdefab') = '{key_split('x-abcdefab')}' # docstring says: ignores hex")