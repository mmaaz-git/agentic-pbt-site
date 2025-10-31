import numpy as np
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

# First test the simple reproduction case
print("=" * 60)
print("REPRODUCTION TEST - Simple case")
print("=" * 60)

matrix = sp.csr_matrix([[5.0, 0.0], [0.0, 3.0]])
print(f"Original nnz: {matrix.nnz}")
print(f"Original data: {matrix.data}")

result = matrix * 0
print(f"After multiply by 0:")
print(f"  nnz: {result.nnz}")
print(f"  data: {result.data}")
print(f"  All values zero? {np.all(result.toarray() == 0)}")

print(f"Expected (according to bug report): nnz should be 0")
print(f"Actual: nnz is {result.nnz} with stored zeros: {result.data}")

# Test the property-based test
print("\n" + "=" * 60)
print("PROPERTY-BASED TEST")
print("=" * 60)

@st.composite
def sparse_matrices(draw):
    rows = draw(st.integers(min_value=1, max_value=20))
    cols = draw(st.integers(min_value=1, max_value=20))
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    dense_array = np.random.rand(rows, cols)
    dense_array[dense_array > density] = 0
    return sp.csr_matrix(dense_array)

failures = []
successes = 0

@given(sparse_matrices())
@settings(max_examples=50)  # Reduced from 300 for faster testing
def test_multiply_zero_should_eliminate_stored_values(matrix):
    global failures, successes
    if matrix.nnz == 0:
        return

    result = matrix * 0
    try:
        assert result.nnz == 0, f"Expected nnz=0, got nnz={result.nnz}"
        assert np.all(result.toarray() == 0)
        successes += 1
    except AssertionError as e:
        failures.append({
            'original_nnz': matrix.nnz,
            'result_nnz': result.nnz,
            'error': str(e)
        })

try:
    test_multiply_zero_should_eliminate_stored_values()
except Exception as e:
    print(f"Test failed with: {e}")

print(f"Failures: {len(failures)}")
print(f"Successes: {successes}")

if failures:
    print(f"\nFirst failure details:")
    print(f"  Original nnz: {failures[0]['original_nnz']}")
    print(f"  Result nnz: {failures[0]['result_nnz']}")
    print(f"  Error: {failures[0]['error']}")

# Test with simple 1x1 matrix as mentioned in bug report
print("\n" + "=" * 60)
print("MINIMAL CASE - 1x1 matrix")
print("=" * 60)

simple_matrix = sp.csr_matrix([[1.0]])
print(f"Original: nnz={simple_matrix.nnz}, data={simple_matrix.data}")
result = simple_matrix * 0
print(f"After *0: nnz={result.nnz}, data={result.data}")
print(f"Matrix is all zeros? {np.all(result.toarray() == 0)}")

# Test what happens with eliminate_zeros
print("\n" + "=" * 60)
print("TESTING eliminate_zeros() AFTER MULTIPLICATION")
print("=" * 60)

matrix2 = sp.csr_matrix([[5.0, 0.0], [0.0, 3.0]])
result2 = matrix2 * 0
print(f"Before eliminate_zeros: nnz={result2.nnz}, data={result2.data}")
result2.eliminate_zeros()
print(f"After eliminate_zeros: nnz={result2.nnz}, data={result2.data}")

# Test different sparse formats
print("\n" + "=" * 60)
print("TESTING DIFFERENT SPARSE FORMATS")
print("=" * 60)

test_data = [[5.0, 0.0], [0.0, 3.0]]
for format_name, format_class in [('CSR', sp.csr_matrix),
                                   ('CSC', sp.csc_matrix),
                                   ('COO', sp.coo_matrix),
                                   ('DOK', sp.dok_matrix),
                                   ('LIL', sp.lil_matrix)]:
    mat = format_class(test_data)
    result = mat * 0
    print(f"{format_name}: original nnz={mat.nnz}, after *0 nnz={result.nnz}")