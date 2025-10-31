from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst
import numpy as np
from pandas.arrays import SparseArray

@given(
    data=npst.arrays(
        dtype=npst.integer_dtypes() | npst.floating_dtypes(),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=100)
    )
)
@settings(max_examples=1000)
def test_density_property(data):
    sparse = SparseArray(data)

    if len(sparse) == 0:
        try:
            density = sparse.density
            assert not np.isnan(density), (
                f"BUG: density={density} for empty array (length=0). "
                f"Should return 0.0 or raise informative error."
            )
            assert not np.isinf(density), f"density should not be Inf for empty array"
        except ZeroDivisionError as e:
            print(f"Got ZeroDivisionError for empty array: {e}")
            raise AssertionError(f"BUG: Got ZeroDivisionError for empty array instead of NaN or 0.0")
    else:
        expected_density = sparse.npoints / len(sparse)
        assert sparse.density == expected_density

# Run the test
if __name__ == "__main__":
    test_density_property()