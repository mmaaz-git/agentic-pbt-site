import pandas.arrays as pa
from hypothesis import given, strategies as st, settings
import traceback

@st.composite
def sparse_arrays(draw):
    size = draw(st.integers(min_value=0, max_value=50))
    fill_value = draw(st.integers(min_value=-10, max_value=10))
    data = draw(st.lists(st.integers(min_value=-100, max_value=100), min_size=size, max_size=size))
    return pa.SparseArray(data, fill_value=fill_value)

@given(sparse_arrays())
@settings(max_examples=10, deadline=1000)
def test_sparse_array_cumsum_length(arr):
    try:
        result = arr.cumsum()
        assert len(result) == len(arr)
        print(f"✓ Passed for array with fill_value={arr.fill_value}, length={len(arr)}")
    except RecursionError:
        print(f"✗ RecursionError for array with fill_value={arr.fill_value}, length={len(arr)}")
        raise
    except Exception as e:
        print(f"✗ Other error for array with fill_value={arr.fill_value}: {e}")
        raise

# Run the test
try:
    test_sparse_array_cumsum_length()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed: {e}")
    traceback.print_exc()