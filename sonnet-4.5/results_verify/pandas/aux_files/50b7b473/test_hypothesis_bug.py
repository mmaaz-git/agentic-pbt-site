#!/usr/bin/env python3
"""Test the Hypothesis property-based test from the bug report"""

from hypothesis import given, strategies as st, settings
from pandas.plotting._matplotlib.converter import TimeFormatter


@given(st.floats(min_value=86400, max_value=172800, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_timeformatter_wraps_at_24_hours(seconds_since_midnight):
    formatter = TimeFormatter(locs=[])
    result = formatter(seconds_since_midnight)

    s = int(seconds_since_midnight)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    _, h = divmod(h, 24)

    assert 0 <= h < 24


if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test...")
    try:
        test_timeformatter_wraps_at_24_hours()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")

    # Test the specific failing input
    print("\nTesting specific failing input: 86400.99999999997")
    try:
        formatter = TimeFormatter(locs=[])
        x = 86400.99999999997
        result = formatter(x)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed with error: {type(e).__name__}: {e}")