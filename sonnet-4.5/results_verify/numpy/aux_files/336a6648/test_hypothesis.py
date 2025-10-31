from pandas.arrays import SparseArray
from hypothesis import given, strategies as st, settings
import sys

@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50)
)
@settings(max_examples=500)
def test_cumsum_length(data):
    arr = SparseArray(data)
    cumsum_result = arr.cumsum()
    assert len(cumsum_result) == len(arr)

if __name__ == "__main__":
    try:
        test_cumsum_length()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)