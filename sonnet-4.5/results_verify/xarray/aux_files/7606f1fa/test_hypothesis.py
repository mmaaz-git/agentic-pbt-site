from hypothesis import given, settings, strategies as st
import numpy as np
from xarray.namedarray.core import NamedArray


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=10)  # Reduced for quicker testing
def test_permute_dims_with_missing_dim_ignore(n):
    data = np.arange(n * 2).reshape(n, 2)
    arr = NamedArray(('x', 'y'), data)

    try:
        result = arr.permute_dims('x', 'z', missing_dims='ignore')
        print(f"n={n}: Success - result.dims = {result.dims}")
        assert result.dims == ('x', 'y'), f"Expected dims ('x', 'y'), got {result.dims}"
    except ValueError as e:
        print(f"n={n}: FAILED - ValueError raised: {e}")
        raise AssertionError(f"ValueError raised despite missing_dims='ignore': {e}")

print("Running Hypothesis test...")
print("-" * 50)
test_permute_dims_with_missing_dim_ignore()