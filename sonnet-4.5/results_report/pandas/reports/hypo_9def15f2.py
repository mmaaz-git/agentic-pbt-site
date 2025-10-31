from hypothesis import given, strategies as st
from pandas.io.excel._base import inspect_excel_format
import pytest


def test_inspect_excel_format_empty_raises():
    """Test that inspect_excel_format raises ValueError on empty input as documented."""
    with pytest.raises(ValueError, match="stream is empty"):
        inspect_excel_format(b'')


if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for inspect_excel_format with empty bytes...")
    print()

    try:
        test_inspect_excel_format_empty_raises()
        print("Test PASSED: ValueError was raised as expected")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        if "DID NOT RAISE" in str(e):
            print(f"Test FAILED: Function did not raise ValueError as expected")
            print(f"Error details: {e}")
        else:
            print(f"Test ERROR: {type(e).__name__}: {e}")