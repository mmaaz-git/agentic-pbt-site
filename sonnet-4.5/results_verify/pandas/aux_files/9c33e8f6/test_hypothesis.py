from hypothesis import given, strategies as st, settings
import pandas.core.arrays.sparse as sparse
import numpy as np
import sys

@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
    fill_value=st.integers(min_value=-10, max_value=10).filter(lambda x: x not in [np.nan])
)
@settings(max_examples=10, deadline=None)
def test_sparse_array_cumsum_should_not_crash(data, fill_value):
    """cumsum should work on sparse arrays with non-null fill values"""
    np_data = np.array(data)
    sparse_arr = sparse.SparseArray(np_data, fill_value=fill_value)

    try:
        result = sparse_arr.cumsum()
        assert len(result) == len(sparse_arr)
        print(f"✓ Passed: data={data[:5]}..., fill_value={fill_value}")
    except RecursionError:
        print(f"✗ RecursionError: data={data[:5]}..., fill_value={fill_value}")
        raise
    except Exception as e:
        print(f"✗ Other error ({type(e).__name__}): data={data[:5]}..., fill_value={fill_value}")
        raise

print("Running Hypothesis property-based test...")
try:
    test_sparse_array_cumsum_should_not_crash()
    print("\nAll tests passed!")
except RecursionError:
    print("\nTest failed with RecursionError as reported")
    sys.exit(1)
except Exception as e:
    print(f"\nTest failed with: {type(e).__name__}: {e}")
    sys.exit(1)