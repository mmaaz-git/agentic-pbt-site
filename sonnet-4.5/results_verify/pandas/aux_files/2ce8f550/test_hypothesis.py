from hypothesis import given, strategies as st
import pandas.core.algorithms as algorithms

@given(st.lists(st.text(), min_size=1))
def test_factorize_round_trip_strings(values):
    """Round-trip property: uniques.take(codes) should equal values"""
    codes, uniques = algorithms.factorize(values)
    reconstructed = uniques.take(codes)
    assert len(reconstructed) == len(values)
    for i, (orig, recon) in enumerate(zip(values, reconstructed)):
        assert orig == recon, f"Mismatch at index {i}: {orig!r} != {recon!r}"

# Test with the specific failing input
if __name__ == "__main__":
    try:
        values = ['', '\x00']
        codes, uniques = algorithms.factorize(values)
        reconstructed = uniques.take(codes)
        assert len(reconstructed) == len(values)
        for i, (orig, recon) in enumerate(zip(values, reconstructed)):
            assert orig == recon, f"Mismatch at index {i}: {orig!r} != {recon!r}"
        print("Test passed with ['', '\\x00']")
    except AssertionError as e:
        print(f"Test failed with ['', '\\x00']: {e}")