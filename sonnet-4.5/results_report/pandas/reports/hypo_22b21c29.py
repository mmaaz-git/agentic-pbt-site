import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
import pandas.core.array_algos.quantile as quantile_module


@given(
    values=npst.arrays(
        dtype=npst.integer_dtypes(endianness='='),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=100),
    ),
)
@settings(max_examples=300)
def test_quantile_integer_array(values):
    qs = np.array([0.0, 0.5, 1.0])
    interpolation = 'linear'

    result = quantile_module.quantile_compat(values, qs, interpolation)

    assert len(result) == len(qs)
    assert result[0] <= result[1] <= result[2], f"Non-monotonic quantiles: {result} for input {values}"


if __name__ == "__main__":
    test_quantile_integer_array()