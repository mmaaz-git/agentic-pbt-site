from hypothesis import given, strategies as st, settings
import pandas.api.types as pat
import re

@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_consistency(pattern):
    """Test from the bug report"""
    try:
        is_compilable = pat.is_re_compilable(pattern)
    except Exception as e:
        print(f"\nException raised by is_re_compilable for pattern {repr(pattern)}: {type(e).__name__}: {e}")
        raise

    if is_compilable:
        try:
            re.compile(pattern)
        except:
            assert False, f"is_re_compilable returned True but re.compile failed for pattern {repr(pattern)}"

if __name__ == "__main__":
    test_is_re_compilable_consistency()