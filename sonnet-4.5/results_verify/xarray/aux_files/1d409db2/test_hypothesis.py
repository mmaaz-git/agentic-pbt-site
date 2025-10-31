import numpy as np
from hypothesis import given, strategies as st, assume, settings
from xarray.indexes.range_index import RangeCoordinateTransform


@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    size=st.integers(min_value=1, max_value=1000),
    position=st.integers(min_value=0, max_value=999)
)
@settings(max_examples=200)
def test_range_transform_forward_reverse_roundtrip(start, stop, size, position):
    assume(position < size)

    transform = RangeCoordinateTransform(
        start=start,
        stop=stop,
        size=size,
        coord_name="x",
        dim="x",
    )

    original_labels = {transform.coord_name: np.array([start + position * transform.step])}
    positions = transform.reverse(original_labels)
    reconstructed_labels = transform.forward(positions)

    assert np.allclose(
        original_labels[transform.coord_name],
        reconstructed_labels[transform.coord_name],
        rtol=1e-10,
        atol=1e-10
    )

# Test with the failing input
if __name__ == "__main__":
    print("Testing with the reported failing input: start=0.0, stop=0.0, size=1, position=0")
    try:
        test_range_transform_forward_reverse_roundtrip(start=0.0, stop=0.0, size=1, position=0)
        print("Test passed")
    except Exception as e:
        print(f"Test failed with error: {e}")

    print("\nRunning hypothesis tests...")
    test_range_transform_forward_reverse_roundtrip()