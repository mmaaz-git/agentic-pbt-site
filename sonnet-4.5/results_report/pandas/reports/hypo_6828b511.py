import re
from hypothesis import given, strategies as st, settings
from pandas.api.types import is_re_compilable


@given(st.one_of(
    st.text(),
    st.binary(),
    st.integers(),
    st.floats(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers())
))
@settings(max_examples=1000)
def test_is_re_compilable_correctness(obj):
    result = is_re_compilable(obj)

    if result:
        try:
            re.compile(obj)
        except TypeError:
            assert False, f"is_re_compilable({obj!r}) returned True but re.compile() raised TypeError"
    else:
        try:
            re.compile(obj)
            assert False, f"is_re_compilable({obj!r}) returned False but re.compile() succeeded"
        except TypeError:
            pass

if __name__ == "__main__":
    test_is_re_compilable_correctness()