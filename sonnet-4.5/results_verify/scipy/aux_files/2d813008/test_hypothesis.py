from hypothesis import given, strategies as st, assume
import numpy as np
from scipy import sparse

@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-100, max_value=100),
                  min_size=1, max_size=100),
    rows=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=1, max_size=100),
    cols=st.lists(st.integers(min_value=0, max_value=99),
                  min_size=1, max_size=100)
)
def test_canonical_format_after_tocsr_tocoo(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = np.array(data[:min_len])
    rows = rows[:min_len]
    cols = cols[:min_len]

    assume(min_len > 0)

    shape = (100, 100)
    original = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    converted = original.tocsr().tocoo()

    coords = list(zip(converted.row.tolist(), converted.col.tolist()))
    has_duplicates = len(coords) != len(set(coords))

    if not has_duplicates:
        assert converted.has_canonical_format, \
            f"COO from CSR should have canonical format when no duplicates. Data: {data[:5]}, rows: {rows[:5]}, cols: {cols[:5]}"

# Run the test with the specific failing case mentioned
print("Testing specific failing case from bug report:")
data = [0.0]
rows = [0]
cols = [0]

try:
    test_canonical_format_after_tocsr_tocoo(data, rows, cols)
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Run a few more hypothesis tests
print("\nRunning hypothesis tests...")
try:
    test_canonical_format_after_tocsr_tocoo()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")