from hypothesis import given, strategies as st, settings
import pandas.io.clipboard as clipboard

@given(
    initial=st.text(),
    changed=st.one_of(st.none(), st.integers(), st.booleans(), st.lists(st.text()))
)
@settings(max_examples=10, deadline=None)  # Reduced examples for faster testing
def test_waitForNewPaste_should_wait_for_string(initial, changed):
    if initial == changed:
        return

    original_paste = clipboard.paste
    call_count = [0]

    def mock_paste():
        call_count[0] += 1
        return initial if call_count[0] == 1 else changed

    clipboard.paste = mock_paste

    try:
        result = clipboard.waitForNewPaste(timeout=0.1)
        assert isinstance(result, str), \
            f"waitForNewPaste should return string, not {type(result).__name__}"
    finally:
        clipboard.paste = original_paste

# Run the test
print("Running property-based test...")
test_waitForNewPaste_should_wait_for_string()
print("Test completed without assertions - checking specific failure case...")

# Test specific failure case
initial = "text"
changed = None

original_paste = clipboard.paste
call_count = [0]

def mock_paste():
    call_count[0] += 1
    return initial if call_count[0] == 1 else changed

clipboard.paste = mock_paste

try:
    result = clipboard.waitForNewPaste(timeout=0.1)
    print(f"Result for initial='{initial}', changed={changed}: {repr(result)}, Type: {type(result).__name__}")
    assert isinstance(result, str), f"waitForNewPaste should return string, not {type(result).__name__}"
except AssertionError as e:
    print(f"AssertionError: {e}")
finally:
    clipboard.paste = original_paste