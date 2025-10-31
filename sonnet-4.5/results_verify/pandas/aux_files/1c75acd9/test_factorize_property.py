import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
import pandas.core.algorithms as algorithms

@given(st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=0, max_size=100))
@settings(max_examples=500)
def test_factorize_round_trip(values):
    """Property: uniques[codes] should reconstruct original values (ignoring NaN sentinel)"""
    try:
        codes, uniques = algorithms.factorize(values)

        reconstructed = []
        for code in codes:
            if code == -1:
                reconstructed.append(np.nan)
            else:
                reconstructed.append(uniques[code])

        for i, (orig, recon) in enumerate(zip(values, reconstructed)):
            if pd.isna(orig) and pd.isna(recon):
                continue
            assert orig == recon, f"Mismatch at index {i}: {orig} != {recon}"
    except (TypeError, ValueError):
        pass

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with specific failing input: ['', '\\x00']")
    try:
        test_factorize_round_trip(['', '\x00'])
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run the full property test
    print("\nRunning property-based test...")
    test_factorize_round_trip()