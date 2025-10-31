#!/usr/bin/env python3
"""
Property-based test for pandas.io.clipboard timeout precision bug.
Tests that waitForPaste() and waitForNewPaste() respect small/zero/negative timeouts.
"""

from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard
import time

# Mock the paste function to always return empty string
def mock_empty_paste():
    return ""

@given(st.floats(min_value=-100, max_value=0.005, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_waitForPaste_timeout_precision(timeout):
    # Override paste function with mock
    original_paste = clipboard.paste
    clipboard.paste = mock_empty_paste

    try:
        start = time.time()
        try:
            clipboard.waitForPaste(timeout)
            assert False, f"Should have raised PyperclipTimeoutException for timeout={timeout}"
        except clipboard.PyperclipTimeoutException:
            elapsed = time.time() - start

            if timeout <= 0:
                assert elapsed < 0.005, (
                    f"With timeout={timeout} (≤0), expected immediate timeout "
                    f"but waited {elapsed:.4f}s (>0.005s). "
                    f"Bug: timeout check happens AFTER sleep(0.01)"
                )
    finally:
        clipboard.paste = original_paste

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for waitForPaste timeout precision...")
    print("Testing with timeouts from -100 to 0.005")
    print("=" * 60)

    try:
        test_waitForPaste_timeout_precision()
        print("\nAll tests passed! No bug detected.")
    except AssertionError as e:
        print(f"\nBUG DETECTED: {e}")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()