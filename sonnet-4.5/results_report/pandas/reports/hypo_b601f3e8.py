import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops
from hypothesis import given, strategies as st, settings


@given(left_val=st.booleans())
@settings(max_examples=10)
def test_kleene_xor_na_commutativity_full(left_val):
    left_with_value = np.array([left_val])
    mask_with_value = np.array([False])

    left_with_na = np.array([False])
    mask_with_na = np.array([True])

    result_value_na, mask_value_na = ops.kleene_xor(left_with_value, libmissing.NA, mask_with_value, None)
    result_na_value, mask_na_value = ops.kleene_xor(left_with_na, left_val, mask_with_na, None)

    assert mask_value_na[0] == mask_na_value[0]
    assert result_value_na[0] == result_na_value[0]


if __name__ == "__main__":
    test_kleene_xor_na_commutativity_full()