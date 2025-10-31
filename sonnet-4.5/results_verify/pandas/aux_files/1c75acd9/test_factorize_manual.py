import numpy as np
import pandas as pd
import pandas.core.algorithms as algorithms

def test_factorize_round_trip_manual(values):
    """Manually test the round-trip property for factorize"""
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

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with specific failing input: ['', '\\x00']")
    try:
        test_factorize_round_trip_manual(['', '\x00'])
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")