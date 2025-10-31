import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad


@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_lpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    assert result == "", f"LPAD({text!r}, {length}, {fill_text!r}) should return empty string for negative length, got {result!r}"


@given(st.text(min_size=1), st.integers(max_value=-1), st.text(min_size=1))
@settings(max_examples=100)
def test_rpad_negative_length_returns_empty(text, length, fill_text):
    result = _sqlite_rpad(text, length, fill_text)
    assert result == "", f"RPAD({text!r}, {length}, {fill_text!r}) should return empty string for negative length, got {result!r}"


if __name__ == "__main__":
    print("Running property-based tests for LPAD/RPAD negative length handling...")
    print()

    print("Testing LPAD with negative lengths:")
    test_lpad_negative_length_returns_empty()

    print("Testing RPAD with negative lengths:")
    test_rpad_negative_length_returns_empty()