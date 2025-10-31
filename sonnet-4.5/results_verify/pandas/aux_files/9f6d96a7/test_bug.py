#!/usr/bin/env python3
"""Test the reported bug"""

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import validate_argsort_with_ascending

# Test case from bug report
print("Testing direct call with None:")
try:
    result = validate_argsort_with_ascending(None, (), {})
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with various inputs:")
# Test with True (should work)
try:
    result = validate_argsort_with_ascending(True, (), {})
    print(f"True: {result}")
except Exception as e:
    print(f"True error: {e}")

# Test with False (should work)
try:
    result = validate_argsort_with_ascending(False, (), {})
    print(f"False: {result}")
except Exception as e:
    print(f"False error: {e}")

# Test with -1 (numpy default axis value)
try:
    result = validate_argsort_with_ascending(-1, (), {})
    print(f"-1 (default axis): {result}")
except Exception as e:
    print(f"-1 error: {e}")

# Test with 0 (valid axis)
try:
    result = validate_argsort_with_ascending(0, (), {})
    print(f"0 (axis=0): {result}")
except Exception as e:
    print(f"0 error: {e}")

# Manual test without hypothesis decorator
print("\nManual tests:")
def test_manual(ascending):
    args = ()
    kwargs = {}
    try:
        result = validate_argsort_with_ascending(ascending, args, kwargs)
        print(f"  Input {ascending}: {result}")
        assert result is True
    except Exception as e:
        print(f"  Input {ascending} failed: {e}")
        return False
    return True

# Run a few examples
test_manual(None)
test_manual(0)
test_manual(-1)
test_manual(1)