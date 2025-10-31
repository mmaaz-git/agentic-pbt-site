#!/usr/bin/env python3
"""Testing the reported bug with pandas.api.types.is_re_compilable"""

import re
from hypothesis import given, strategies as st, settings
import pandas.api.types as pat

# First, test with the property-based test
@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_should_not_crash(s):
    try:
        result = pat.is_re_compilable(s)
        assert isinstance(result, bool), f"Should return bool, got {type(result)}"
        print(f"✓ Input {s!r}: returned {result}")
    except Exception as e:
        print(f"✗ is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}")
        raise AssertionError(
            f"is_re_compilable crashed on input {s!r} with {type(e).__name__}: {e}"
        )

print("Running property-based test...")
try:
    test_is_re_compilable_should_not_crash()
    print("Property-based test completed without crashes")
except AssertionError as e:
    print(f"Property-based test failed: {e}")

# Now test the specific failing example
print("\n" + "="*50)
print("Testing specific example: ')'")
print("="*50)

try:
    result = pat.is_re_compilable(')')
    print(f"Result: {result}")
except Exception as e:
    print(f"Crashed with {type(e).__name__}: {e}")

# Test other invalid regex patterns mentioned
invalid_patterns = ['[', '(', '?', '*', '(?', '(*', '[[]', '\\']
print("\n" + "="*50)
print("Testing other invalid regex patterns:")
print("="*50)

for pattern in invalid_patterns:
    try:
        result = pat.is_re_compilable(pattern)
        print(f"Pattern {pattern!r}: returned {result}")
    except Exception as e:
        print(f"Pattern {pattern!r}: crashed with {type(e).__name__}: {e}")

# Test valid patterns to ensure they work
valid_patterns = ['.*', 'hello', '[a-z]+', '\\d+', '(abc)+']
print("\n" + "="*50)
print("Testing valid regex patterns (should return True):")
print("="*50)

for pattern in valid_patterns:
    try:
        result = pat.is_re_compilable(pattern)
        print(f"Pattern {pattern!r}: returned {result}")
    except Exception as e:
        print(f"Pattern {pattern!r}: crashed with {type(e).__name__}: {e}")

# Test non-string inputs (should return False)
non_strings = [1, 1.5, None, [], {}, object()]
print("\n" + "="*50)
print("Testing non-string inputs (should return False):")
print("="*50)

for obj in non_strings:
    try:
        result = pat.is_re_compilable(obj)
        print(f"Object {obj!r}: returned {result}")
    except Exception as e:
        print(f"Object {obj!r}: crashed with {type(e).__name__}: {e}")