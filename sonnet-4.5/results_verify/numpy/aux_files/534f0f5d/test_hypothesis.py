import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

@given(st.just(ma.array([], dtype=int)))
@settings(max_examples=5)
def test_flatnotmasked_edges_empty_array(arr):
    result = ma.flatnotmasked_edges(arr)
    if result is not None:
        assert len(result) == 0 or (len(result) == 2 and result[0] >= 0 and result[1] >= result[0])

# Run the test
test_flatnotmasked_edges_empty_array()