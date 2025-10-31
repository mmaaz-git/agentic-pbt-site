import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

# First, let's run the simple reproduction case
print("=" * 60)
print("Simple Reproduction Test")
print("=" * 60)

data = np.array([1.0, 2.0, 3.0])
row = np.array([0, 0, 0])
col = np.array([0, 0, 1])
A = sp.coo_matrix((data, (row, col)), shape=(2, 2))

print(f"Initial data array: {A.data}")
print(f"Initial row array: {A.row}")
print(f"Initial col array: {A.col}")
print(f"Before sum_duplicates: has_canonical_format = {A.has_canonical_format}")

A.sum_duplicates()
print(f"\nAfter sum_duplicates: has_canonical_format = {A.has_canonical_format}")
print(f"Data array after sum_duplicates: {A.data}")
print(f"Row array after sum_duplicates: {A.row}")
print(f"Col array after sum_duplicates: {A.col}")

# Store original data for comparison
original_data = A.data.copy()

# Modify data directly
A.data[0] = 999.0
print(f"\nAfter data modification: has_canonical_format = {A.has_canonical_format}")
print(f"Data array after modification: {A.data}")
print(f"Original data was: {original_data}")
print(f"Data actually changed: {not np.array_equal(A.data, original_data)}")

# Now let's test the property-based test
print("\n" + "=" * 60)
print("Property-Based Test")
print("=" * 60)

@st.composite
def coo_matrices_with_duplicates(draw):
    n = draw(st.integers(min_value=2, max_value=15))
    size = draw(st.integers(min_value=2, max_value=30))
    data = draw(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                         min_size=size, max_size=size))
    row = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=size, max_size=size))
    col = draw(st.lists(st.integers(min_value=0, max_value=n-1), min_size=size, max_size=size))
    return sp.coo_matrix((data, (row, col)), shape=(n, n))

@given(coo_matrices_with_duplicates())
@settings(max_examples=50)
def test_canonical_format_flag_invalidation(A):
    A.sum_duplicates()
    assert A.has_canonical_format

    original_data = A.data.copy()
    if len(A.data) > 0:  # Only modify if there's data
        A.data[0] = A.data[0] + 1

    if A.has_canonical_format and not np.array_equal(A.data, original_data):
        raise AssertionError("BUG: has_canonical_format flag not invalidated after data modification")

try:
    test_canonical_format_flag_invalidation()
    print("Property-based test completed without issues")
except AssertionError as e:
    print(f"Property-based test failed: {e}")
except Exception as e:
    print(f"Property-based test encountered an error: {e}")

# Let's also test modifying row and col arrays
print("\n" + "=" * 60)
print("Testing row/col array modification")
print("=" * 60)

data2 = np.array([1.0, 2.0, 3.0])
row2 = np.array([0, 1, 1])
col2 = np.array([0, 0, 1])
B = sp.coo_matrix((data2, (row2, col2)), shape=(3, 3))

B.sum_duplicates()
print(f"After sum_duplicates: has_canonical_format = {B.has_canonical_format}")
print(f"Row array: {B.row}")
print(f"Col array: {B.col}")

# Try modifying row array
if len(B.row) > 0:
    B.row[0] = 2
    print(f"After modifying row[0] to 2: has_canonical_format = {B.has_canonical_format}")
    print(f"Row array: {B.row}")

# Create another matrix and test col modification
C = sp.coo_matrix((data2, (row2, col2)), shape=(3, 3))
C.sum_duplicates()
if len(C.col) > 0:
    old_col = C.col[0]
    C.col[0] = 2
    print(f"\nAfter modifying col[0] from {old_col} to 2: has_canonical_format = {C.has_canonical_format}")
    print(f"Col array: {C.col}")