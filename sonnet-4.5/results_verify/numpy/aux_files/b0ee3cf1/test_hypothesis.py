import numpy as np
from hypothesis import given, strategies as st

@given(
    base_type=st.sampled_from(['i4', 'f8', 'c16']),
    shape_size=st.integers(min_value=1, max_value=5)
)
def test_dtype_descr_round_trip_with_shapes(base_type, shape_size):
    shape = tuple(range(1, shape_size + 1))
    dtype = np.dtype((base_type, shape))

    descr = np.lib.format.dtype_to_descr(dtype)
    restored = np.lib.format.descr_to_dtype(descr)

    assert restored == dtype, \
        f"Round-trip failed: {dtype} -> {descr} -> {restored}"

# Run the test
if __name__ == "__main__":
    test_dtype_descr_round_trip_with_shapes()