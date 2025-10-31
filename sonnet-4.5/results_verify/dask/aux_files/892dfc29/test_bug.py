#!/usr/bin/env python3
"""Test script to reproduce the format_bytes bug report."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.utils
from hypothesis import given, strategies as st, settings, example

# First, let's test the specific failing example from the bug report
def test_specific_example():
    n = 1125894277343089729
    result = dask.utils.format_bytes(n)

    print(f"n = {n}")
    print(f"n < 2**60: {n < 2**60}")
    print(f"format_bytes({n}) = '{result}'")
    print(f"Length: {len(result)}")
    print()

    assert n < 2**60, f"Test value {n} is not less than 2**60"

    if len(result) > 10:
        print(f"❌ BUG CONFIRMED: Output length {len(result)} exceeds claimed maximum of 10")
        return False
    else:
        print(f"✓ Output length {len(result)} is within claimed maximum of 10")
        return True

# Property-based test from the bug report
@given(st.integers(min_value=1, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    result = dask.utils.format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10 (per docstring)"

# Let's explore the boundary cases more systematically
def test_boundary_cases():
    print("Testing boundary cases around PiB values:")
    print()

    # Test values around 1 PiB (2**50)
    test_values = [
        (2**50 - 1, "Just below 1 PiB"),
        (2**50, "Exactly 1 PiB"),
        (900 * 2**50 // 1000, "0.9 PiB"),
        (999 * 2**50 // 1000, "0.999 PiB"),
        (1000 * 2**50 // 1000, "1 PiB"),
        (1001 * 2**50 // 1000, "1.001 PiB"),
        (9999 * 2**50 // 10000, "0.9999 PiB"),
        (10000 * 2**50 // 10000, "1 PiB exact"),
    ]

    # Values that will produce >= 1000 PiB (but still < 2**60)
    for multiplier in [1000, 1001, 1010, 1100, 1234]:
        if multiplier * 2**50 < 2**60:
            test_values.append((multiplier * 2**50, f"{multiplier} PiB"))

    violations = []
    for n, description in test_values:
        if n >= 2**60:
            continue  # Skip values >= 2**60

        result = dask.utils.format_bytes(n)
        length = len(result)

        print(f"{description:20} n={n:20} -> '{result:12}' (len={length})")

        if length > 10:
            violations.append((n, result, length, description))
            print(f"    ❌ VIOLATION: Length {length} > 10")

    print()
    return violations

# Let's check what happens at the actual implementation level
def analyze_implementation():
    print("Analyzing implementation behavior:")
    print()

    # The implementation uses n >= k * 0.9 for deciding which unit to use
    # For PiB: k = 2**50
    # So values >= 0.9 * 2**50 will be formatted as PiB

    threshold = int(0.9 * 2**50)
    print(f"PiB threshold (0.9 * 2**50) = {threshold}")

    # Test values that will produce different length outputs
    test_cases = [
        (threshold - 1, "Just below PiB threshold"),
        (threshold, "At PiB threshold"),
        (int(9.99 * 2**50), "9.99 PiB"),
        (int(10.0 * 2**50), "10.0 PiB"),
        (int(99.99 * 2**50), "99.99 PiB"),
        (int(100.0 * 2**50), "100.0 PiB"),
        (int(999.99 * 2**50), "999.99 PiB"),
        (int(1000.0 * 2**50), "1000.0 PiB"),
    ]

    for n, desc in test_cases:
        if n >= 2**60:
            print(f"Skipping {desc} - exceeds 2**60")
            continue

        result = dask.utils.format_bytes(n)
        value_in_pib = n / 2**50
        print(f"{desc:30} -> '{result}' (len={len(result)}) [{value_in_pib:.2f} PiB]")

if __name__ == "__main__":
    print("="*60)
    print("Testing dask.utils.format_bytes bug report")
    print("="*60)
    print()

    # Test the specific example
    print("1. Testing specific example from bug report:")
    print("-" * 40)
    bug_exists = not test_specific_example()

    # Test boundary cases
    print("\n2. Testing boundary cases:")
    print("-" * 40)
    violations = test_boundary_cases()

    # Analyze implementation
    print("\n3. Implementation analysis:")
    print("-" * 40)
    analyze_implementation()

    # Try the hypothesis test
    print("\n4. Running property-based test (may find violations):")
    print("-" * 40)
    try:
        test_format_bytes_length_claim()
        print("Property test passed for 1000 examples")
    except AssertionError as e:
        print(f"Property test failed: {e}")
        bug_exists = True

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    if bug_exists or violations:
        print("❌ BUG CONFIRMED: The docstring claim is violated")
        print(f"   Found {len(violations)} violations in boundary tests")
        print("   The docstring states 'For all values < 2**60, the output is always <= 10 characters'")
        print("   But values >= 1000 * 2**50 produce outputs with 11+ characters")
    else:
        print("✓ No bug found in tests")