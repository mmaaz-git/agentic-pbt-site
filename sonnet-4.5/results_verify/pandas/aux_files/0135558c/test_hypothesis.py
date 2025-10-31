from hypothesis import given, strategies as st, settings
from pandas.arrays import SparseArray

@st.composite
def sparse_arrays(draw, min_size=0, max_size=100):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    dtype_choice = draw(st.sampled_from(['int64', 'float64', 'bool']))

    if dtype_choice == 'int64':
        values = draw(st.lists(st.integers(min_value=-1000, max_value=1000),
                              min_size=size, max_size=size))
        fill_value = 0
    elif dtype_choice == 'float64':
        values = draw(st.lists(st.floats(min_value=-1e6, max_value=1e6,
                                        allow_nan=False, allow_infinity=False),
                              min_size=size, max_size=size))
        fill_value = 0.0
    else:
        values = draw(st.lists(st.booleans(), min_size=size, max_size=size))
        fill_value = False

    kind = draw(st.sampled_from(['integer', 'block']))
    return SparseArray(values, fill_value=fill_value, kind=kind)

@given(sparse_arrays())
@settings(max_examples=100)
def test_density_in_range(arr):
    """Density should always be between 0 and 1"""
    try:
        density = arr.density
        assert 0 <= density <= 1, f"Density {density} not in [0, 1]"
    except ZeroDivisionError as e:
        print(f"Failed on array with length {len(arr)}: {e}")
        raise

if __name__ == "__main__":
    test_density_in_range()