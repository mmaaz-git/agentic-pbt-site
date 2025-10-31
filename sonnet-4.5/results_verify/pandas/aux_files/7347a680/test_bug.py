#!/usr/bin/env python3
"""Test script to reproduce the reported bug."""

import numpy as np
import traceback
from pandas.core.interchange.from_dataframe import parse_datetime_format_str

def test_valid_timezone():
    """Test with a valid timezone."""
    print("Testing with valid timezone 'UTC'...")
    data = np.array([0, 1000, 2000], dtype=np.int64)
    format_str = "tss:UTC"
    try:
        result = parse_datetime_format_str(format_str, data)
        print(f"Success! Result type: {type(result)}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

def test_invalid_timezone_0():
    """Test with invalid timezone '0' as reported."""
    print("\nTesting with invalid timezone '0'...")
    data = np.array([0, 1000, 2000], dtype=np.int64)
    format_str = "tss:0"
    try:
        result = parse_datetime_format_str(format_str, data)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

def test_invalid_timezone_x80():
    """Test with invalid timezone '\x80' as reported."""
    print("\nTesting with invalid timezone '\\x80'...")
    data = np.array([0, 1000, 2000], dtype=np.int64)
    format_str = "tss:\x80"
    try:
        result = parse_datetime_format_str(format_str, data)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

def test_invalid_timezone_random():
    """Test with another invalid timezone."""
    print("\nTesting with invalid timezone 'NotATimezone'...")
    data = np.array([0, 1000, 2000], dtype=np.int64)
    format_str = "tss:NotATimezone"
    try:
        result = parse_datetime_format_str(format_str, data)
        print(f"Success! Result: {result}")
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

def test_empty_timezone():
    """Test with empty timezone (should work)."""
    print("\nTesting with empty timezone ''...")
    data = np.array([0, 1000, 2000], dtype=np.int64)
    format_str = "tss:"
    try:
        result = parse_datetime_format_str(format_str, data)
        print(f"Success! Result type: {type(result)}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

def test_hypothesis_property():
    """Test the hypothesis property-based test."""
    print("\nRunning Hypothesis property-based test...")
    try:
        from hypothesis import given, strategies as st

        @given(
            resolution=st.sampled_from(['s', 'm', 'u', 'n']),
            tz=st.text(min_size=1, max_size=20)
        )
        def test_parse_datetime_format_str_handles_invalid_tz(resolution, tz):
            format_str = f"ts{resolution}:{tz}"
            data = np.array([0, 1000, 2000], dtype=np.int64)
            result = parse_datetime_format_str(format_str, data)

        # Run a few examples
        test_parse_datetime_format_str_handles_invalid_tz()
        print("Hypothesis test completed (would fail on many inputs)")
    except ImportError:
        print("Hypothesis not installed, skipping property-based test")
    except Exception as e:
        print(f"Hypothesis test failed as expected: {type(e).__name__}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing parse_datetime_format_str timezone handling")
    print("=" * 60)

    test_valid_timezone()
    test_invalid_timezone_0()
    test_invalid_timezone_x80()
    test_invalid_timezone_random()
    test_empty_timezone()
    test_hypothesis_property()

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Valid timezones (UTC, empty string) work correctly")
    print("- Invalid timezones ('0', '\\x80', 'NotATimezone') raise pytz.exceptions.UnknownTimeZoneError")
    print("- The error message is not very helpful for users of the interchange protocol")
    print("=" * 60)