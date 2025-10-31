import pandas as pd
import pandas.core.algorithms as algorithms

# Test with a pandas Series to avoid deprecation warning
values = pd.Series(['', '\x00'])
codes, uniques = algorithms.factorize(values)

print(f"Input: {repr(values.tolist())}")
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")

for i, code in enumerate(codes):
    recon = uniques[code]
    orig = values.iloc[i]
    print(f"Index {i}: orig={repr(orig)}, recon={repr(recon)}, match={orig == recon}")

# Also test that empty string and null char are indeed different
print(f"\n'' == '\\x00': {'' == '\x00'}")
print(f"Empty string length: {len('')}")
print(f"Null char string length: {len('\x00')}")