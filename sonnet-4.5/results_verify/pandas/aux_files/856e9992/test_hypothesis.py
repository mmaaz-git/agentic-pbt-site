from hypothesis import given, strategies as st, settings
from Cython.Utility import pylong_join, _pylong_join


@given(st.integers(min_value=-10, max_value=20))
@settings(max_examples=1000)
def test_pylong_join_consistency(count):
    result1 = pylong_join(count)
    result2 = _pylong_join(count)

    assert result1 == result2, (
        f"Inconsistency between pylong_join and _pylong_join for count={count}:\n"
        f"  pylong_join:  {result1!r}\n"
        f"  _pylong_join: {result2!r}"
    )

if __name__ == "__main__":
    test_pylong_join_consistency()