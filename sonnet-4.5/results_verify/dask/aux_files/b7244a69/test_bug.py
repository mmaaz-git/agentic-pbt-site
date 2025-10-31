#!/usr/bin/env python3
"""Test the reported bug between format_time and parse_timedelta"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from dask.utils import format_time, parse_timedelta
from hypothesis import given, strategies as st, settings, example
import traceback

# First, let's test the specific examples from the bug report
print("=" * 60)
print("Testing specific examples from bug report:")
print("=" * 60)

test_values = [601, 3600, 7200, 86400]

for t in test_values:
    formatted = format_time(t)
    print(f"format_time({t}) = '{formatted}'")
    try:
        parsed = parse_timedelta(formatted)
        print(f"  Parsed successfully: {parsed}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Now test values that should work (smaller values)
print("\n" + "=" * 60)
print("Testing smaller values that should work:")
print("=" * 60)

small_values = [0.001, 0.1, 1, 10, 100, 599]
for t in small_values:
    formatted = format_time(t)
    print(f"format_time({t}) = '{formatted}'")
    try:
        parsed = parse_timedelta(formatted)
        print(f"  Parsed successfully: {parsed}")
        # Check round-trip
        if abs(parsed - t) / t > 0.05:
            print(f"  WARNING: Round-trip error > 5%: {abs(parsed - t) / t * 100:.2f}%")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Now run the property test from the bug report
print("\n" + "=" * 60)
print("Running property-based test:")
print("=" * 60)

failures = []

@given(st.floats(min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False))
@example(601.0)  # Force test of the failing case
@settings(max_examples=50)  # Reduced for testing
def test_format_parse_time_roundtrip(t):
    """Property: format_time produces parseable output"""
    formatted = format_time(t)
    try:
        parsed = parse_timedelta(formatted)
        rel_error = abs(parsed - t) / t
        if rel_error >= 0.05:
            failures.append((t, formatted, parsed, rel_error))
            return  # Don't assert to continue testing
    except Exception as e:
        failures.append((t, formatted, str(e), None))
        return  # Don't assert to continue testing

# Run the test
try:
    test_format_parse_time_roundtrip()
    print(f"Property test completed. Found {len(failures)} failures out of tests run.")
except Exception as e:
    print(f"Property test failed with exception: {e}")
    traceback.print_exc()

if failures:
    print(f"\nFirst 10 failures:")
    for i, failure in enumerate(failures[:10]):
        if len(failure) == 4 and failure[3] is not None:
            t, formatted, parsed, rel_error = failure
            print(f"  {i+1}. t={t:.2f}, formatted='{formatted}', parsed={parsed}, rel_error={rel_error:.2%}")
        else:
            t, formatted, error, _ = failure
            print(f"  {i+1}. t={t:.2f}, formatted='{formatted}', error='{error}'")

# Test edge cases around the boundaries
print("\n" + "=" * 60)
print("Testing boundary values:")
print("=" * 60)

boundary_values = [599, 600, 601, 7199, 7200, 7201, 172799, 172800, 172801]
for t in boundary_values:
    formatted = format_time(t)
    print(f"format_time({t}) = '{formatted}'")
    try:
        parsed = parse_timedelta(formatted)
        print(f"  Parsed successfully: {parsed}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Test what parse_timedelta CAN handle
print("\n" + "=" * 60)
print("Testing what parse_timedelta CAN parse:")
print("=" * 60)

test_strings = [
    "10m",
    "1s",
    "10m1s",  # Without space
    "10m 1s",  # With space (as format_time produces)
    "24hr",
    "0m",
    "24hr0m",  # Without space
    "24hr 0m",  # With space (as format_time produces)
    "1.23 ms",
    "123.45 us",
]

for s in test_strings:
    print(f"parse_timedelta('{s}'): ", end="")
    try:
        result = parse_timedelta(s)
        print(f"{result}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")