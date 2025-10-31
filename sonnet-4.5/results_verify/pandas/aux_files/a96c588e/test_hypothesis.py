import numpy as np
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import process_skipna


@given(
    skipna=st.one_of(st.booleans(), st.none(), st.from_type(np.ndarray)),
    args=st.tuples(st.integers(), st.text())
)
def test_process_skipna_returns_python_bool(skipna, args):
    result_skipna, result_args = process_skipna(skipna, args)
    assert isinstance(result_skipna, bool) and not isinstance(result_skipna, np.bool_)

# Run test
if __name__ == "__main__":
    import traceback
    try:
        test_process_skipna_returns_python_bool()
        print("All tests passed")
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

    # Try specific failing case mentioned
    print("\nTrying specific failing case: np.bool_(True)")
    try:
        test_func = test_process_skipna_returns_python_bool.hypothesis.inner_test
        test_func(skipna=np.bool_(True), args=(1, "test"))
        print("Test passed for np.bool_(True)")
    except AssertionError as e:
        print(f"Test failed for np.bool_(True): {e}")