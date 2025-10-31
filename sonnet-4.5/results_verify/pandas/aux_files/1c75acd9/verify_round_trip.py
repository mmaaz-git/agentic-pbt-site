import pandas as pd
import numpy as np
import pandas.core.algorithms as algorithms

# Test the round-trip property as documented
values = pd.Series(['', '\x00'])
codes, uniques = algorithms.factorize(values)

print(f"Original values: {repr(values.tolist())}")
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")

# Use take as documented
reconstructed = uniques.take(codes)
print(f"Reconstructed via uniques.take(codes): {reconstructed}")
print(f"Reconstructed as list: {repr(reconstructed.tolist())}")

# Check if round-trip works
print(f"\nRound-trip check:")
for i in range(len(values)):
    orig = values.iloc[i]
    recon = reconstructed[i]
    print(f"  values[{i}] = {repr(orig)}, reconstructed[{i}] = {repr(recon)}, equal = {orig == recon}")

# Verify these are different values
print(f"\nVerifying empty string and null char are different:")
print(f"  '' == '\\x00': {'' == '\x00'}")
print(f"  ord(''): N/A (empty)")
print(f"  ord('\\x00'): {ord('\x00')}")