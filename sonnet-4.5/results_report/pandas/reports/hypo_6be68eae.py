from hypothesis import given, strategies as st, settings, Phase, example
import sys
import traceback


def paste_klipper_mock(stdout_bytes):
    """Mock of the actual paste_klipper implementation from pandas.io.clipboard"""
    ENCODING = 'utf-8'
    clipboardContents = stdout_bytes.decode(ENCODING)

    # These assertions are from lines 277-279 of pandas/io/clipboard/__init__.py
    assert len(clipboardContents) > 0
    assert clipboardContents.endswith("\n")

    if clipboardContents.endswith("\n"):
        clipboardContents = clipboardContents[:-1]
    return clipboardContents


@given(st.binary())
@example(b"")  # Empty string - triggers first assertion
@example(b"Hello")  # No newline - triggers second assertion
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate])
def test_paste_klipper_handles_arbitrary_bytes(data):
    """
    Property: paste_klipper should handle any valid UTF-8 clipboard data
    without crashing due to assertions.
    """
    try:
        decoded = data.decode('utf-8')
    except UnicodeDecodeError:
        return  # Skip invalid UTF-8

    # Try to call the function - it may fail with AssertionError
    try:
        result = paste_klipper_mock(data)
        # If successful, verify the result is correct
        assert result == decoded[:-1] if decoded.endswith('\n') else decoded
    except AssertionError as e:
        # Report failing case
        print(f"\nFalsifying example: {repr(data)}")
        print(f"Decoded string: {repr(decoded)}")
        print(f"Error: {e}")
        traceback.print_exc()
        # Re-raise to fail the test
        raise

if __name__ == "__main__":
    print("Running Hypothesis test to find assertion failures...")
    try:
        test_paste_klipper_handles_arbitrary_bytes()
        print("\nAll tests passed (should not happen - assertions should fail)")
    except AssertionError:
        print("\nTest failed as expected - found inputs that trigger assertions")