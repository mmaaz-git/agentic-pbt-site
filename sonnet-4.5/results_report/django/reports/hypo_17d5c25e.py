from hypothesis import given, strategies as st, example
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad


@given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
@example(text="hello", length=10)
def test_lpad_with_empty_fill_text_violates_length(text, length):
    fill_text = ""
    result = _sqlite_lpad(text, length, fill_text)

    if result is not None:
        assert len(result) == length, (
            f"lpad with empty fill_text should still return exact length. "
            f"Expected length {length}, got {len(result)}: {result!r}"
        )


@given(st.text(min_size=0, max_size=50), st.integers(min_value=0, max_value=100))
@example(text="hello", length=10)
def test_rpad_with_empty_fill_text_violates_length(text, length):
    fill_text = ""
    result = _sqlite_rpad(text, length, fill_text)

    if result is not None:
        assert len(result) == length, (
            f"rpad with empty fill_text should still return exact length. "
            f"Expected length {length}, got {len(result)}: {result!r}"
        )


if __name__ == "__main__":
    print("Running Hypothesis tests for LPAD/RPAD with empty fill_text...")
    print("=" * 60)

    try:
        test_lpad_with_empty_fill_text_violates_length()
        print("LPAD test passed")
    except AssertionError as e:
        print(f"LPAD test FAILED: {e}")

    try:
        test_rpad_with_empty_fill_text_violates_length()
        print("RPAD test passed")
    except AssertionError as e:
        print(f"RPAD test FAILED: {e}")