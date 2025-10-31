import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@settings(max_examples=500)
@given(
    n=st.integers(min_value=2, max_value=15),
    density=st.floats(min_value=0.1, max_value=0.5)
)
def test_coo_from_csr_canonical_property(n, density):
    coo = sp.random(n, n, density=density, format='coo', random_state=42)
    coo.sum_duplicates()

    csr = coo.tocsr()
    coo_from_csr = csr.tocoo()

    positions = list(zip(coo_from_csr.row, coo_from_csr.col))
    unique_positions = set(positions)
    has_duplicates = len(positions) != len(unique_positions)

    assert not has_duplicates, "COO from CSR should have no duplicates"
    assert coo_from_csr.has_canonical_format, \
        "COO from CSR should have has_canonical_format=True"

# Run the test
if __name__ == "__main__":
    test_coo_from_csr_canonical_property()
    print("Test completed")