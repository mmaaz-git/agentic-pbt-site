import pandas as pd
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.compat.pdcompat import timestamp_as_unit, default_precision_timestamp

print("Testing direct reproduction of the bug...")
print("=" * 60)

try:
    print("1. Testing timestamp_as_unit with year=1000:")
    ts = pd.Timestamp(year=1000, month=1, day=1)
    print(f"   Created timestamp: {ts}")
    print(f"   Timestamp unit: {ts.unit}")
    result = timestamp_as_unit(ts, 'ns')
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

print()

try:
    print("2. Testing default_precision_timestamp with year=1000:")
    result = default_precision_timestamp(year=1000, month=1, day=1)
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("Testing with property-based tests...")
print()

@given(
    st.integers(min_value=1, max_value=9999),
    st.integers(min_value=1, max_value=12),
    st.integers(min_value=1, max_value=28),
    st.sampled_from(['s', 'ms', 'us', 'ns'])
)
def test_timestamp_as_unit_preserves_value(year, month, day, unit):
    ts = pd.Timestamp(year=year, month=month, day=day)
    result = timestamp_as_unit(ts, unit)
    assert result.year == ts.year
    assert result.month == ts.month
    assert result.day == ts.day


@given(
    st.integers(min_value=1, max_value=9999),
    st.integers(min_value=1, max_value=12),
    st.integers(min_value=1, max_value=28)
)
def test_default_precision_timestamp_returns_ns(year, month, day):
    result = default_precision_timestamp(year=year, month=month, day=day)
    assert result.unit == "ns"

try:
    print("Running test_timestamp_as_unit_preserves_value...")
    test_timestamp_as_unit_preserves_value()
    print("Test passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")
except Exception as e:
    print(f"Test failed with error: {type(e).__name__}: {e}")

print()

try:
    print("Running test_default_precision_timestamp_returns_ns...")
    test_default_precision_timestamp_returns_ns()
    print("Test passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")
except Exception as e:
    print(f"Test failed with error: {type(e).__name__}: {e}")

# Test some edge cases
print()
print("=" * 60)
print("Testing edge cases...")
print()

test_cases = [
    (1, 1, 1),
    (100, 1, 1),
    (1677, 1, 1),  # Just before ns range
    (1678, 1, 1),  # Start of ns range
    (2262, 1, 1),  # End of ns range
    (2263, 1, 1),  # Just after ns range
    (9999, 1, 1),
]

for year, month, day in test_cases:
    print(f"Testing year={year}:")
    try:
        ts = pd.Timestamp(year=year, month=month, day=day)
        print(f"  Created timestamp: {ts}, unit={ts.unit}")

        # Try timestamp_as_unit
        try:
            result = timestamp_as_unit(ts, 'ns')
            print(f"  timestamp_as_unit(ts, 'ns'): Success - {result}")
        except Exception as e:
            print(f"  timestamp_as_unit(ts, 'ns'): {type(e).__name__}")

        # Try default_precision_timestamp
        try:
            result = default_precision_timestamp(year=year, month=month, day=day)
            print(f"  default_precision_timestamp: Success - {result}, unit={result.unit}")
        except Exception as e:
            print(f"  default_precision_timestamp: {type(e).__name__}")
    except Exception as e:
        print(f"  ERROR creating timestamp: {type(e).__name__}: {e}")
    print()