from hypothesis import given, strategies as st
import pytest


def paste_klipper_mock(stdout_bytes):
    """Mock of the actual paste_klipper implementation"""
    ENCODING = 'utf-8'
    clipboardContents = stdout_bytes.decode(ENCODING)

    assert len(clipboardContents) > 0
    assert clipboardContents.endswith("\n")
    if clipboardContents.endswith("\n"):
        clipboardContents = clipboardContents[:-1]
    return clipboardContents


@given(st.binary())
def test_paste_klipper_handles_arbitrary_bytes(data):
    """
    Property: paste_klipper should handle any valid UTF-8 clipboard data
    without crashing, but it doesn't.
    """
    try:
        decoded = data.decode('utf-8')
    except UnicodeDecodeError:
        return  # Skip invalid UTF-8

    if len(decoded) == 0 or not decoded.endswith('\n'):
        # These cases cause assertion failures
        with pytest.raises(AssertionError):
            paste_klipper_mock(data)
    else:
        # Should work fine with data ending in newline
        result = paste_klipper_mock(data)
        assert result == decoded[:-1]  # Should strip the trailing newline


if __name__ == "__main__":
    # Run the test
    test_paste_klipper_handles_arbitrary_bytes()
    print("Property-based test completed successfully")