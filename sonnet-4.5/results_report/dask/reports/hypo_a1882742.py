from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote

@given(st.lists(st.tuples(st.text(min_size=1, max_size=5), st.integers()), max_size=10))
def test_unquote_dict_correct_format(items):
    task = (dict, items)
    result = unquote(task)
    assert result == dict(items)

if __name__ == "__main__":
    # Run the test
    test_unquote_dict_correct_format()