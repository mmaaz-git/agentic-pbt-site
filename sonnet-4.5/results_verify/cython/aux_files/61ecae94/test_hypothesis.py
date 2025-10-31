from hypothesis import given, strategies as st, example
from Cython.TestUtils import _parse_pattern

@given(st.text())
@example("/start")  # Crashes: no closing slash
@example("/")  # Crashes: empty after slash
@example(":/end")  # Crashes: end marker without closing slash
def test_parse_pattern_no_crash(pattern):
    """Property: _parse_pattern should not crash on any string input"""
    try:
        result = _parse_pattern(pattern)
        assert isinstance(result, tuple)
        assert len(result) == 3
        print(f"✓ Pattern '{pattern[:50]}...' returned: {result}")
    except Exception as e:
        print(f"✗ Pattern '{pattern[:50]}...' crashed with: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    test_parse_pattern_no_crash()