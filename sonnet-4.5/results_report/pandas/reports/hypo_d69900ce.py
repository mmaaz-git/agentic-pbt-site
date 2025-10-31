from hypothesis import given, strategies as st, settings, example
import pandas.io.clipboard as clipboard

@given(
    initial=st.text(),
    changed=st.one_of(st.none(), st.integers(), st.booleans(), st.lists(st.text(), max_size=3), st.dictionaries(st.text(), st.text(), max_size=3))
)
@example(initial="text", changed=None)  # Specific example that fails
@settings(max_examples=100, deadline=None)
def test_waitForNewPaste_should_wait_for_string(initial, changed):
    """Test that waitForNewPaste should only return strings, not other types"""

    # Skip if the initial and changed are the same (no change detected)
    if initial == changed:
        return

    # Save original paste function
    original_paste = clipboard.paste
    call_count = [0]

    def mock_paste():
        call_count[0] += 1
        return initial if call_count[0] == 1 else changed

    clipboard.paste = mock_paste

    try:
        result = clipboard.waitForNewPaste(timeout=0.1)

        # The function docstring says it waits for "a new text string"
        # and "returns this text" - so it should always return a string
        assert isinstance(result, str), \
            f"waitForNewPaste should return string, not {type(result).__name__}. " \
            f"Returned: {repr(result)}"

    finally:
        clipboard.paste = original_paste

if __name__ == "__main__":
    # Run the test
    test_waitForNewPaste_should_wait_for_string()