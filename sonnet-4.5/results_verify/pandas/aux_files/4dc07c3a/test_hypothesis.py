#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.core.interchange.from_dataframe import parse_datetime_format_str


@given(st.integers(min_value=2**62, max_value=2**63-1))
@settings(max_examples=100)
def test_parse_datetime_days_overflow(days):
    format_str = "tdD"
    data = np.array([days], dtype=np.int64)

    result = parse_datetime_format_str(format_str, data)

    expected_seconds = np.uint64(days) * np.uint64(24 * 60 * 60)

    if expected_seconds > 2**63 - 1:
        result_as_int = result.view('int64')[0]
        assert result_as_int >= 0 or days < 0, \
            f"Silent overflow: positive days={days} produced negative result={result_as_int}"

# Run the test
if __name__ == "__main__":
    import traceback

    print("Running hypothesis test...")
    print("=" * 60)

    try:
        test_parse_datetime_days_overflow()
        print("Test passed! No failures found.")
    except AssertionError as e:
        print(f"Test FAILED with assertion error:")
        print(str(e))
        traceback.print_exc()
    except Exception as e:
        print(f"Test encountered unexpected error:")
        print(str(e))
        traceback.print_exc()