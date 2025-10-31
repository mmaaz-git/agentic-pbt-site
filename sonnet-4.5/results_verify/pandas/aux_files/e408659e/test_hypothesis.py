import pandas as pd
from pandas.api.types import infer_dtype
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=500)
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=20))
def test_infer_dtype_skipna_consistency_floats(float_values):
    assume(len(float_values) > 0)

    result_without_none = infer_dtype(float_values, skipna=True)

    float_values_with_none = float_values + [None]
    result_with_none = infer_dtype(float_values_with_none, skipna=True)

    assert result_without_none == result_with_none, \
        f"infer_dtype with skipna=True should ignore None: " \
        f"{float_values} -> {result_without_none}, " \
        f"{float_values_with_none} -> {result_with_none}"

if __name__ == "__main__":
    test_infer_dtype_skipna_consistency_floats()