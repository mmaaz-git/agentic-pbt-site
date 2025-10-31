from hypothesis import given, strategies as st
import dask.base
import sys
import traceback

@given(st.one_of(
    st.text(min_size=1),
    st.binary(min_size=1),
    st.tuples(st.text(min_size=1), st.integers()),
))
def test_key_split_returns_string(s):
    result = dask.base.key_split(s)
    assert isinstance(result, str)

# Run the test
if __name__ == "__main__":
    print("Testing with Hypothesis...")
    try:
        test_key_split_returns_string()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed with exception: {e}")
        traceback.print_exc()