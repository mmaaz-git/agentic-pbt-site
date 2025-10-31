#!/usr/bin/env python3

# Test the bug reported for parse_list

print("Testing parse_list bug...")

# First, let's test the basic reproducer
from Cython.Build.Dependencies import parse_list

print("\nTest 1: Single double-quote")
try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\nTest 2: Single single-quote")
try:
    result = parse_list("'")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Let's also test some valid cases to understand expected behavior
print("\nTest 3: Empty string")
try:
    result = parse_list("")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 4: Normal string")
try:
    result = parse_list("a")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTest 5: String with proper quotes")
try:
    result = parse_list('"hello"')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Now test with hypothesis
print("\n\n=== Testing with Hypothesis ===")
from hypothesis import given, strategies as st, settings, seed
import traceback

@given(st.text())
@settings(max_examples=1000, print_blob=True)
def test_parse_list_returns_list(s):
    try:
        result = parse_list(s)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
    except Exception as e:
        print(f"\nFailed on input: {repr(s)}")
        print(f"Exception: {type(e).__name__}: {e}")
        raise

print("Running hypothesis test...")
try:
    test_parse_list_returns_list()
    print("All tests passed!")
except Exception as e:
    print(f"Hypothesis test failed!")
    traceback.print_exc()