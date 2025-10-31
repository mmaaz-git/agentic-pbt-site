from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@given(value=st.one_of(st.none(), st.just(0), st.just(False), st.just([]), st.just({})))
@settings(max_examples=50, deadline=None)
def test_waitForPaste_should_wait_for_string(value):
    original_paste = clipboard.paste
    clipboard.paste = lambda: value

    try:
        result = clipboard.waitForPaste(timeout=0.1)
        assert isinstance(result, str) and result != "", \
            f"waitForPaste should wait for non-empty string, not {type(value).__name__}"
    except clipboard.PyperclipTimeoutException:
        pass
    finally:
        clipboard.paste = original_paste

# Run the test
if __name__ == "__main__":
    test_waitForPaste_should_wait_for_string()
    print("Test completed")