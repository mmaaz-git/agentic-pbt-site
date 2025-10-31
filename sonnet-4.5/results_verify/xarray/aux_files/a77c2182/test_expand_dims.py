import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.namedarray.core import NamedArray


@st.composite
def namedarray_with_potential_collision(draw):
    ndim = draw(st.integers(min_value=1, max_value=4))
    dim_choices = [f"dim_{i}" for i in range(10)]
    dims = draw(st.lists(st.sampled_from(dim_choices), min_size=ndim, max_size=ndim, unique=True))
    shape = tuple(2 for _ in range(ndim))
    data = np.ones(shape)
    return NamedArray(tuple(dims), data)


@given(namedarray_with_potential_collision())
@settings(max_examples=500)
def test_expand_dims_no_default_duplicates(arr):
    expanded = arr.expand_dims()
    assert len(expanded.dims) == len(set(expanded.dims)), \
        f"expand_dims() created duplicate dimensions: {expanded.dims}"

if __name__ == "__main__":
    test_expand_dims_no_default_duplicates()
    print("Test completed")