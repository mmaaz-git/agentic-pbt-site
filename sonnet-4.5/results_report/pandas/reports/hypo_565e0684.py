#!/usr/bin/env python3
"""
Property-based test for pandas Klipper clipboard implementation using Hypothesis.
"""
from hypothesis import given, settings, strategies as st, example
import pandas.io.clipboard as clipboard
import sys
import traceback


@given(st.text())
@settings(max_examples=1000)
@example('\x00')  # Specific example that should fail
def test_klipper_copy_handles_all_text(text):
    try:
        clipboard.set_clipboard("klipper")
    except (clipboard.PyperclipException, FileNotFoundError, ImportError) as e:
        print(f"Skipping test: Klipper not available - {e}")
        return

    try:
        clipboard.copy(text)
        result = clipboard.paste()
        assert result == text or result == text + "\n", f"Expected {repr(text)} or {repr(text + '\n')}, got {repr(result)}"
        print(f"✓ Test passed for text: {repr(text)}")
    except Exception as e:
        print(f"✗ Test failed for text: {repr(text)}")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the test using Hypothesis
    print("Running property-based test with Hypothesis...")
    print("Testing specifically with null byte character '\\x00'...")
    try:
        test_klipper_copy_handles_all_text()
    except Exception as e:
        print(f"\nTest failed!")
        sys.exit(1)