from hypothesis import given, strategies as st, assume, example
import numpy as np
from scipy import sparse

def check_canonical_format_after_conversion(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = np.array(data[:min_len])
    rows = rows[:min_len]
    cols = cols[:min_len]

    if min_len == 0:
        return True

    shape = (100, 100)
    original = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    converted = original.tocsr().tocoo()

    coords = list(zip(converted.row.tolist(), converted.col.tolist()))
    has_duplicates = len(coords) != len(set(coords))

    if not has_duplicates:
        return converted.has_canonical_format
    return True  # If there are duplicates, we don't expect canonical format

# Test the specific failing case
print("Testing specific failing case from bug report:")
data = [0.0]
rows = [0]
cols = [0]

result = check_canonical_format_after_conversion(data, rows, cols)
if result:
    print("Test passed: has_canonical_format is True")
else:
    print("Test failed: has_canonical_format is False when it should be True")
    print("This confirms the bug!")

print("\nTesting with value 1.0:")
data = [1.0]
result = check_canonical_format_after_conversion(data, rows, cols)
if result:
    print("Test passed: has_canonical_format is True")
else:
    print("Test failed: has_canonical_format is False when it should be True")

print("\nTesting multiple unique elements:")
data = [1.0, 2.0, 3.0]
rows = [0, 1, 2]
cols = [0, 1, 2]
result = check_canonical_format_after_conversion(data, rows, cols)
if result:
    print("Test passed: has_canonical_format is True")
else:
    print("Test failed: has_canonical_format is False when it should be True")

print("\nTesting with duplicates that get summed:")
data = [1.0, 2.0, 3.0]
rows = [0, 0, 1]
cols = [0, 0, 1]
result = check_canonical_format_after_conversion(data, rows, cols)
print(f"Result for duplicates case: {result} (doesn't matter for this test)")

# Now run the hypothesis test
@given(
    data=st.lists(st.floats(allow_nan=False, allow_infinity=False,
                           min_value=-100, max_value=100),
                  min_size=1, max_size=10),
    rows=st.lists(st.integers(min_value=0, max_value=9),
                  min_size=1, max_size=10),
    cols=st.lists(st.integers(min_value=0, max_value=9),
                  min_size=1, max_size=10)
)
@example(data=[0.0], rows=[0], cols=[0])
@example(data=[1.0], rows=[0], cols=[0])
def test_canonical_format(data, rows, cols):
    min_len = min(len(data), len(rows), len(cols))
    data = np.array(data[:min_len])
    rows = rows[:min_len]
    cols = cols[:min_len]

    assume(min_len > 0)

    shape = (10, 10)
    original = sparse.coo_matrix((data, (rows, cols)), shape=shape)
    converted = original.tocsr().tocoo()

    coords = list(zip(converted.row.tolist(), converted.col.tolist()))
    has_duplicates = len(coords) != len(set(coords))

    if not has_duplicates:
        assert converted.has_canonical_format, \
            f"COO from CSR should have canonical format when no duplicates"

print("\n" + "=" * 60)
print("Running hypothesis tests...")
try:
    test_canonical_format()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed - Bug confirmed!")
    print(f"Error: {e}")