from hypothesis import given, strategies as st
import dask.base

@given(st.one_of(
    st.text(min_size=1),
    st.binary(min_size=1),
    st.tuples(st.text(min_size=1), st.integers()),
))
def test_key_split_returns_string(s):
    result = dask.base.key_split(s)
    assert isinstance(result, str)

# Run the test
test_key_split_returns_string()