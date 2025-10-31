from pandas.core import ops
from hypothesis import given, strategies as st
import sys

# Test with hypothesis
@given(st.booleans(), st.booleans())
def test_kleene_and_without_mask_equals_regular_and(a, b):
    try:
        result = ops.kleene_and(a, b, None, None)
        expected = a and b
        assert result == expected
        print(f"Success: kleene_and({a}, {b}) = {result}, expected {expected}")
    except RecursionError as e:
        print(f"RecursionError with kleene_and({a}, {b}): {str(e)[:100]}")
        raise
    except Exception as e:
        print(f"Other error with kleene_and({a}, {b}): {e}")
        raise

# Try to run the test
print("Running hypothesis test...")
try:
    test_kleene_and_without_mask_equals_regular_and()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Manual reproduction
print("\n--- Manual reproduction ---")
print("Testing: ops.kleene_and(False, True, None, None)")
sys.setrecursionlimit(100)  # Set a lower limit to catch recursion quickly
try:
    result = ops.kleene_and(False, True, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:200]}")
except Exception as e:
    print(f"Other error: {e}")

print("\nTesting: ops.kleene_or(False, True, None, None)")
try:
    result = ops.kleene_or(False, True, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:200]}")
except Exception as e:
    print(f"Other error: {e}")

print("\nTesting: ops.kleene_xor(False, True, None, None)")
try:
    result = ops.kleene_xor(False, True, None, None)
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:200]}")
except Exception as e:
    print(f"Other error: {e}")