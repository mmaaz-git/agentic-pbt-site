import pandas as pd
import pyarrow as pa
from hypothesis import given, strategies as st, settings
import traceback

@st.composite
def empty_or_small_arrays(draw):
    dtype = draw(st.sampled_from([pa.int64(), pa.string()]))
    size = draw(st.integers(min_value=0, max_value=5))

    if size == 0:
        pa_array = pa.array([], type=dtype)
    elif pa.types.is_integer(dtype):
        values = draw(st.lists(
            st.one_of(st.integers(min_value=-100, max_value=100), st.none()),
            min_size=size, max_size=size
        ))
        pa_array = pa.array(values, type=dtype)
    else:
        values = draw(st.lists(
            st.one_of(st.text(max_size=10), st.none()),
            min_size=size, max_size=size
        ))
        pa_array = pa.array(values, type=dtype)

    return pd.array(pa_array, dtype=pd.ArrowDtype(dtype))


@given(empty_or_small_arrays())
@settings(max_examples=10, deadline=None)
def test_empty_array_operations(arr):
    print(f"Testing array with length {len(arr)}, dtype {arr.dtype}")
    if len(arr) == 0:
        try:
            result_empty_take = arr.take([])
            assert len(result_empty_take) == 0
            print("  - Empty array with empty take: PASSED")
        except Exception as e:
            print(f"  - Empty array with empty take: FAILED - {type(e).__name__}: {e}")
            raise
    else:
        try:
            result_empty_take = arr.take([])
            assert len(result_empty_take) == 0
            print("  - Non-empty array with empty take: PASSED")
        except Exception as e:
            print(f"  - Non-empty array with empty take: FAILED - {type(e).__name__}: {e}")
            raise

if __name__ == "__main__":
    print("Running Hypothesis test for ArrowExtensionArray.take([])")
    print("=" * 60)
    try:
        test_empty_array_operations()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        traceback.print_exc()