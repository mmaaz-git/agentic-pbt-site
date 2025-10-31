#!/usr/bin/env python3
"""Test to reproduce the bug reported for ArrowORCEngine._aggregate_files"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine

# First, test the hypothesis test provided in the bug report
@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=200)
def test_aggregate_files_with_none_stripes(split_stripes_val):
    parts = [[("file1.orc", None)], [("file2.orc", None)]]

    result = ArrowORCEngine._aggregate_files(
        aggregate_files=True,
        split_stripes=split_stripes_val,
        parts=parts
    )

    assert result is not None

# Now test the specific failing case
def test_specific_failing_case():
    """Test the specific case mentioned in the bug report"""
    parts = [[("file1.orc", None)], [("file2.orc", None)]]

    try:
        result = ArrowORCEngine._aggregate_files(
            aggregate_files=True,
            split_stripes=2,
            parts=parts
        )
        print("Test passed without error. Result:", result)
        return "NO_ERROR"
    except TypeError as e:
        print(f"TypeError caught: {e}")
        return "TYPE_ERROR"
    except Exception as e:
        print(f"Other error caught: {type(e).__name__}: {e}")
        return "OTHER_ERROR"

if __name__ == "__main__":
    # Test the specific failing case first
    print("Testing specific failing case with split_stripes=2:")
    result = test_specific_failing_case()
    print(f"Result: {result}\n")

    # Now run the hypothesis test
    print("Running hypothesis test:")
    try:
        test_aggregate_files_with_none_stripes()
        print("Hypothesis test completed - no errors found")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        import traceback
        traceback.print_exc()