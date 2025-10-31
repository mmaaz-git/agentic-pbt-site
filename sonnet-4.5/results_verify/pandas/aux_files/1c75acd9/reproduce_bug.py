import pandas.core.algorithms as algorithms

values = ['', '\x00']
codes, uniques = algorithms.factorize(values)

print(f"Input: {repr(values)}")
print(f"Codes: {codes}")
print(f"Uniques: {uniques}")

for i, code in enumerate(codes):
    recon = uniques[code]
    orig = values[i]
    print(f"Index {i}: orig={repr(orig)}, recon={repr(recon)}, match={orig == recon}")