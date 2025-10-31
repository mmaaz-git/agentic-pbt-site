from hypothesis import given, strategies as st
import numpy as np
from scipy.sparse.csgraph._validation import validate_graph

@given(st.sampled_from([np.float32, np.float64, np.int32]))
def test_validate_graph_respects_dtype(dtype):
    G = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    result = validate_graph(G, directed=True, dtype=dtype)

    assert result.dtype == dtype, \
        f"Expected dtype {dtype}, but got {result.dtype}"

# Run the test
test_validate_graph_respects_dtype()