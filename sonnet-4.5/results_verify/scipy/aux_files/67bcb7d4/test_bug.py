import scipy.constants as sc
from hypothesis import given, strategies as st, settings
import pytest

# First, let's test the hypothesis test from the bug report
@given(st.integers())
@settings(max_examples=10)  # Reduced for quick testing
def test_find_with_integer_should_handle_gracefully(num):
    print(f"Testing with num={num}")
    try:
        result = sc.find(num)
        print(f"Result: {result}")
        assert isinstance(result, list)
    except Exception as e:
        print(f"Error with num={num}: {type(e).__name__}: {e}")
        raise

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_find_with_integer_should_handle_gracefully()
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Now test the specific reproduction case
print("\n" + "="*50)
print("Testing specific case: sc.find(123)")
print("="*50)
try:
    result = sc.find(123)
    print(f"Result: {result}")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Test with other non-string types
print("\n" + "="*50)
print("Testing with other non-string types:")
print("="*50)

test_cases = [
    (0, "integer 0"),
    (42, "integer 42"),
    (3.14, "float"),
    ([], "empty list"),
    ([1,2,3], "list"),
    ({}, "empty dict"),
    (True, "boolean True"),
    (False, "boolean False"),
]

for value, description in test_cases:
    print(f"\nTesting with {description}: {value}")
    try:
        result = sc.find(value)
        print(f"  Result: {result}")
    except AttributeError as e:
        print(f"  AttributeError: {e}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")

# Test valid cases
print("\n" + "="*50)
print("Testing valid cases:")
print("="*50)

valid_cases = [
    (None, "None (should return all keys)"),
    ("electron", "string 'electron'"),
    ("", "empty string"),
    ("ELECTRON", "uppercase string"),
]

for value, description in valid_cases:
    print(f"\nTesting with {description}")
    try:
        result = sc.find(value)
        if value is None:
            print(f"  Found {len(result)} constants")
        else:
            print(f"  Found {len(result)} constants matching '{value}'")
            if len(result) <= 5:
                for key in result:
                    print(f"    - {key}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")