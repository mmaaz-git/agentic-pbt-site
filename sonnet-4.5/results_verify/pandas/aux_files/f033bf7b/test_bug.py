#!/usr/bin/env python3
"""Test script to reproduce the _validate_names bug"""

from hypothesis import given, strategies as st, settings
import pandas as pd
import io
import traceback

print("=" * 60)
print("Testing pandas._validate_names behavior with unhashable names")
print("=" * 60)

# Test 1: Hypothesis test as provided in bug report
print("\n1. Running Hypothesis property test:")
print("-" * 40)

@given(
    unhashable_names=st.lists(
        st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=3), st.integers(), min_size=1, max_size=2)
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=10)  # Reduced for quick verification
def test_unhashable_names_should_raise_valueerror(unhashable_names):
    csv_data = ','.join(['0'] * len(unhashable_names)) + '\n'
    try:
        df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
        return "ERROR: No exception raised"
    except TypeError as e:
        return f"BUG CONFIRMED: Got TypeError: {e}"
    except ValueError as e:
        return f"Correct: Got ValueError: {e}"

# Run a few examples manually without hypothesis
test_cases = [
    [[1], [2]],
    [[1, 2], [3, 4]],
    [{"a": 1}, {"b": 2}],
]

for unhashable_names in test_cases:
    csv_data = ','.join(['0'] * len(unhashable_names)) + '\n'
    try:
        df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
        result = "ERROR: No exception raised"
    except TypeError as e:
        result = f"BUG CONFIRMED: Got TypeError: {e}"
    except ValueError as e:
        result = f"Correct: Got ValueError: {e}"

    print(f"Test case: {unhashable_names}")
    print(f"Result: {result}")
    print()

# Test 2: Direct reproduction as in bug report
print("\n2. Direct reproduction test:")
print("-" * 40)

csv_data = "1,2,3\n4,5,6"
names = [[1, 2], [3, 4], [5, 6]]

print(f"CSV data: {repr(csv_data)}")
print(f"Names: {names}")
print()

try:
    df = pd.read_csv(io.StringIO(csv_data), names=names, header=None)
    print("ERROR: No exception was raised!")
except TypeError as e:
    print(f"BUG CONFIRMED: Raised {type(e).__name__}: {e}")
    print("Expected: ValueError")
    print("\nFull traceback:")
    traceback.print_exc()
except ValueError as e:
    print(f"Correct: Raised {type(e).__name__}: {e}")

# Test 3: Direct test of _validate_names function
print("\n3. Direct test of _validate_names function:")
print("-" * 40)

from pandas.io.parsers.readers import _validate_names

test_inputs = [
    ([[1], [2]], "lists as column names"),
    ([{"a": 1}, {"b": 2}], "dicts as column names"),
    ([set([1]), set([2])], "sets as column names"),
]

for names, description in test_inputs:
    print(f"\nTesting {description}: {names}")
    try:
        _validate_names(names)
        print("  No exception raised")
    except TypeError as e:
        print(f"  BUG: TypeError raised: {e}")
    except ValueError as e:
        print(f"  Correct: ValueError raised: {e}")

# Test 4: Check what happens with valid hashable names
print("\n4. Testing with valid hashable names:")
print("-" * 40)

valid_test_cases = [
    (["a", "b", "c"], "strings"),
    ([1, 2, 3], "integers"),
    ([("a", 1), ("b", 2)], "tuples"),
]

for names, description in valid_test_cases:
    print(f"\nTesting {description}: {names}")
    try:
        _validate_names(names)
        print("  Validation passed (expected)")
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")

# Test 5: Check duplicate detection
print("\n5. Testing duplicate detection:")
print("-" * 40)

dup_names = ["a", "b", "a"]
print(f"Testing duplicates: {dup_names}")
try:
    _validate_names(dup_names)
    print("  No exception raised")
except ValueError as e:
    print(f"  Correct: ValueError raised: {e}")
except Exception as e:
    print(f"  Unexpected: {type(e).__name__}: {e}")