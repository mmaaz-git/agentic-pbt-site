from hypothesis import given, strategies as st
import numpy as np
from xarray.core.indexes import PandasMultiIndex
from xarray.core.variable import Variable

@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_pandas_multi_index_stack_unstack_roundtrip(level_a_values):
    level_b_values = ['a', 'b']

    var_a = Variable(("dim_a",), np.array(level_a_values))
    var_b = Variable(("dim_b",), np.array(level_b_values))

    variables = {"level_a": var_a, "level_b": var_b}

    multi_idx = PandasMultiIndex.stack(variables, "stacked")

    unstacked_indexes, pd_multi_index = multi_idx.unstack()

    assert "level_a" in unstacked_indexes
    assert "level_b" in unstacked_indexes

if __name__ == "__main__":
    # Test with the failing input directly
    print("Testing with level_a_values=[0, 0]")
    try:
        test_pandas_multi_index_stack_unstack_roundtrip([0, 0])
        print("Test passed")
    except Exception as e:
        print(f"Test failed with error: {e}")

    # Run the hypothesis test
    print("\nRunning hypothesis test...")
    test_pandas_multi_index_stack_unstack_roundtrip()