from hypothesis import given, settings, strategies as st
import numpy as np
from xarray.namedarray.core import NamedArray


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=100)
def test_permute_dims_with_missing_dim_ignore(n):
    data = np.arange(n * 2).reshape(n, 2)
    arr = NamedArray(('x', 'y'), data)

    result = arr.permute_dims('x', 'z', missing_dims='ignore')

    assert result.dims == ('x', 'y')


if __name__ == "__main__":
    # Run the test
    test_permute_dims_with_missing_dim_ignore()