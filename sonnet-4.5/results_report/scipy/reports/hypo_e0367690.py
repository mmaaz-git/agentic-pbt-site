from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import sparse

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-100, max_value=100),
                  min_size=2, max_size=100),
    rows=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=2, max_size=100),
    cols=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=2, max_size=100)
)
def test_canonical_format_preserved_after_copy(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = list(data[:min_len])
    rows = list(rows[:min_len])
    cols = list(cols[:min_len])

    assume(min_len >= 2)

    # Force duplicate coordinates
    rows[0] = rows[1]
    cols[0] = cols[1]

    shape = (100, 100)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    mat.sum_duplicates()

    copied = mat.copy()

    assert copied.has_canonical_format, \
        f"copy() should preserve has_canonical_format flag. Original: {mat.has_canonical_format}, Copy: {copied.has_canonical_format}"

# Run the test
if __name__ == "__main__":
    test_canonical_format_preserved_after_copy()