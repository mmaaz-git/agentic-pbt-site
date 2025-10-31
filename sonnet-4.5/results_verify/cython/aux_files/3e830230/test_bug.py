#!/usr/bin/env python3

# Test 1: Reproduce the exact bug report
from Cython.Compiler import Options

large_int = -9007199254740993
result = Options.parse_variable_value(str(large_int))

print("=== Bug Report Reproduction ===")
print(f"Input: {large_int}")
print(f"Result: {result}")
print(f"Type: {type(result)}")
print(f"Equal: {result == large_int}")
print()

# Test 2: Test various negative numbers
print("=== Testing Various Negative Numbers ===")
test_values = [
    -1,
    -10,
    -123,
    -999999,
    -9007199254740993,  # The reported case
    -9007199254740992,  # One less
    -9007199254740994,  # One more
]

for val in test_values:
    parsed = Options.parse_variable_value(str(val))
    print(f"Input: {val:20} → Result: {parsed:20} | Type: {type(parsed).__name__:5} | Equal: {parsed == val}")

print()

# Test 3: Test positive large numbers
print("=== Testing Positive Large Numbers ===")
positive_test = [
    1,
    123,
    999999,
    9007199254740993,
    12345678901234567890,
]

for val in positive_test:
    parsed = Options.parse_variable_value(str(val))
    print(f"Input: {val:20} → Result: {parsed:20} | Type: {type(parsed).__name__:5} | Equal: {parsed == val}")

print()

# Test 4: Check float precision boundary
print("=== Float Precision Boundary (2^53) ===")
# Float can exactly represent integers up to 2^53
boundary = 2**53
test_around_boundary = [
    boundary - 2,
    boundary - 1,
    boundary,
    boundary + 1,
    boundary + 2,
]

for val in test_around_boundary:
    for sign in [1, -1]:
        test_val = val * sign
        parsed = Options.parse_variable_value(str(test_val))
        is_equal = parsed == test_val
        print(f"Input: {test_val:20} → Result: {parsed:20} | Equal: {is_equal}")

print()

# Test 5: Run the hypothesis test
print("=== Running Hypothesis Test ===")
try:
    from hypothesis import given, strategies as st

    failures = []

    @given(st.integers())
    def test_parse_variable_value_preserves_integers(n):
        s = str(n)
        result = Options.parse_variable_value(s)
        if s.lstrip('-').isdigit():
            if result != n:
                failures.append((n, result))

    # Run a limited number of tests
    test_parse_variable_value_preserves_integers()

    if failures:
        print(f"Found {len(failures[:10])} failures:")
        for inp, out in failures[:10]:
            print(f"  {inp} → {out}")
    else:
        print("No failures found in hypothesis testing")

except ImportError:
    print("Hypothesis not installed, skipping property-based test")