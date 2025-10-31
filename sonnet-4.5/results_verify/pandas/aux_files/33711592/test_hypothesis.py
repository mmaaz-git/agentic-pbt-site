from hypothesis import given, strategies as st
import re
from pandas.core.dtypes.inference import is_re_compilable


@given(
    pattern=st.one_of(
        st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=10),
        st.integers(),
        st.floats(),
        st.none(),
    )
)
def test_is_re_compilable_consistent_with_re_compile(pattern):
    result = is_re_compilable(pattern)

    try:
        re.compile(pattern)
        can_compile = True
    except (TypeError, re.error):
        can_compile = False

    if can_compile:
        assert result, f"is_re_compilable({pattern!r}) returned False but re.compile succeeded"


if __name__ == "__main__":
    import sys
    try:
        test_is_re_compilable_consistent_with_re_compile()
        print("Hypothesis test passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")
        sys.exit(1)