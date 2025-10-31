#!/usr/bin/env python3
"""Hypothesis test from the bug report."""

from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard

@given(st.sampled_from(["utf-8", "UTF-8", "utf_8", "UTF_8", "Utf-8"]))
@settings(max_examples=10)
def test_clipboard_accepts_python_valid_utf8_encodings(encoding):
    """Test that pandas clipboard functions accept all valid Python UTF-8 encodings."""
    # First verify Python accepts this encoding
    text = "test"
    text.encode(encoding)  # This should always work

    # Test read_clipboard
    try:
        read_clipboard(encoding=encoding)
    except (NotImplementedError, ValueError) as e:
        if "only supports utf-8 encoding" in str(e):
            raise AssertionError(f"read_clipboard rejected valid encoding {encoding}: {e}")
        # Other errors (like clipboard not available) are fine
    except Exception:
        # Non-encoding related errors are fine
        pass

    # Test to_clipboard
    df = pd.DataFrame([[1, 2], [3, 4]])
    try:
        to_clipboard(df, encoding=encoding)
    except (NotImplementedError, ValueError) as e:
        if "only supports utf-8 encoding" in str(e):
            raise AssertionError(f"to_clipboard rejected valid encoding {encoding}: {e}")
        # Other errors (like clipboard not available) are fine
    except Exception:
        # Non-encoding related errors are fine
        pass

if __name__ == "__main__":
    print("Running Hypothesis test...")
    try:
        test_clipboard_accepts_python_valid_utf8_encodings()
        print("Test completed successfully!")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")