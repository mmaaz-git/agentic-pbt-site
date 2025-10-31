from hypothesis import given, strategies as st, settings, assume
from xarray.indexes import RangeIndex
import numpy as np

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200)
def test_linspace_num_one_endpoint(start, stop):
    assume(start != stop)

    index = RangeIndex.linspace(start, stop, num=1, endpoint=True, dim="x")

    transform = index.transform
    coords = transform.forward({transform.dim: np.array([0])})
    values = coords[transform.coord_name]

    assert len(values) == 1

if __name__ == "__main__":
    test_linspace_num_one_endpoint()