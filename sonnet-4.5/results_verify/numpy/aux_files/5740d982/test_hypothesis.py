from hypothesis import given, strategies as st, settings
import numpy.char as char


@settings(max_examples=200)
@given(st.lists(st.text(min_size=1, max_size=5), min_size=1), st.text(min_size=1, max_size=2), st.text(min_size=1, max_size=5))
def test_replace_matches_python(strings, old, new):
    arr = char.array(strings)
    np_replaced = char.replace(arr, old, new)
    for i, s in enumerate(strings):
        py_replaced = s.replace(old, new)
        assert np_replaced[i] == py_replaced, f'{s!r}.replace({old!r}, {new!r}): numpy={np_replaced[i]!r}, python={py_replaced!r}'

if __name__ == "__main__":
    test_replace_matches_python()
    print("All tests passed!")