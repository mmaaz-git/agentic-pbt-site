import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_format_dtdelta


@given(st.floats(min_value=0.1, max_value=1e6), st.floats(min_value=0.1, max_value=1e6))
@settings(max_examples=100)
def test_format_dtdelta_always_returns_string(lhs, rhs):
    for connector in ["+", "-", "*", "/"]:
        result = _sqlite_format_dtdelta(connector, lhs, rhs)
        if result is not None:
            assert isinstance(result, str), f"format_dtdelta({connector!r}, {lhs}, {rhs}) should return string, got {type(result)}"

if __name__ == "__main__":
    test_format_dtdelta_always_returns_string()
    print("Test completed")