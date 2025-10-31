import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas._libs import missing as libmissing


@given(
    rows=st.integers(min_value=2, max_value=5),
    cols=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10)  # Reduced for quicker testing
def test_masked_sum_object_dtype_with_axis(rows, cols):
    print(f"Testing with rows={rows}, cols={cols}")
    values_obj = np.arange(rows * cols).reshape(rows, cols).astype(object)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[0, 0] = True

    try:
        result_axis1_obj = masked_sum(values_obj, mask, skipna=True, axis=1)
        result_axis1_float = masked_sum(values_obj.astype(float), mask, skipna=True, axis=1)

        if result_axis1_obj is not libmissing.NA and result_axis1_float is not libmissing.NA:
            if hasattr(result_axis1_obj, 'shape') and hasattr(result_axis1_float, 'shape'):
                assert result_axis1_obj.shape == result_axis1_float.shape, f"Shape mismatch: {result_axis1_obj.shape} != {result_axis1_float.shape}"
                for i in range(len(result_axis1_obj)):
                    assert result_axis1_obj[i] == result_axis1_float[i], f"Value mismatch at index {i}: {result_axis1_obj[i]} != {result_axis1_float[i]}"
        print(f"  ✓ Passed")
    except Exception as e:
        print(f"  ✗ Failed with error: {type(e).__name__}: {e}")
        raise

# Run the test
test_masked_sum_object_dtype_with_axis()