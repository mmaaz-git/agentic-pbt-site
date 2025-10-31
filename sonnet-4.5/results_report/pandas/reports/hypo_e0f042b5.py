import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op


@settings(max_examples=500)
@given(
    exponent=st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10)
)
def test_masked_arith_op_rpow_base_one_should_return_one(exponent):
    x = np.array([exponent, np.nan, exponent], dtype=object)
    y = 1.0

    result = _masked_arith_op(x, y, roperator.rpow)

    assert result[0] == 1.0, f"Expected 1.0 for 1.0**{exponent}, got {result[0]}"
    assert result[2] == 1.0, f"Expected 1.0 for 1.0**{exponent}, got {result[2]}"

if __name__ == "__main__":
    test_masked_arith_op_rpow_base_one_should_return_one()