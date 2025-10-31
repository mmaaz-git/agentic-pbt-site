"""Simple reproduction of the reset_index bug"""

import pandas as pd
import dask.dataframe as dd

# Test with different partition configurations
print("Testing reset_index behavior with different partitions:\n")

# Test 1: Single partition (should work correctly)
print("Test 1: Single partition")
data = [(0, 0.0), (0, 0.0)]
pdf = pd.DataFrame(data, columns=['a', 'b'])
ddf = dd.from_pandas(pdf, npartitions=1)

reset_pdf = pdf.reset_index(drop=True)
reset_ddf = ddf.reset_index(drop=True).compute()

print(f"  Pandas index: {reset_pdf.index.tolist()}")
print(f"  Dask index:   {reset_ddf.index.tolist()}")
print(f"  Match: {reset_pdf.index.tolist() == reset_ddf.index.tolist()}\n")

# Test 2: Two partitions (bug appears)
print("Test 2: Two partitions (bug case)")
ddf = dd.from_pandas(pdf, npartitions=2)
reset_ddf = ddf.reset_index(drop=True).compute()

print(f"  Pandas index: {reset_pdf.index.tolist()}")
print(f"  Dask index:   {reset_ddf.index.tolist()}")
print(f"  Match: {reset_pdf.index.tolist() == reset_ddf.index.tolist()}\n")

# Test 3: Larger DataFrame with multiple partitions
print("Test 3: Larger DataFrame with 3 partitions")
data = [(i, float(i)) for i in range(6)]
pdf = pd.DataFrame(data, columns=['a', 'b'])
ddf = dd.from_pandas(pdf, npartitions=3)

reset_pdf = pdf.reset_index(drop=True)
reset_ddf = ddf.reset_index(drop=True).compute()

print(f"  Pandas index: {reset_pdf.index.tolist()}")
print(f"  Dask index:   {reset_ddf.index.tolist()}")
print(f"  Match: {reset_pdf.index.tolist() == reset_ddf.index.tolist()}\n")

# Test 4: Test with drop=False (should this also be affected?)
print("Test 4: reset_index with drop=False")
data = [(0, 0.0), (0, 0.0)]
pdf = pd.DataFrame(data, columns=['a', 'b'])
pdf.index = ['x', 'y']  # Custom index
ddf = dd.from_pandas(pdf, npartitions=2)

reset_pdf = pdf.reset_index(drop=False)
reset_ddf = ddf.reset_index(drop=False).compute()

print(f"  Pandas new index: {reset_pdf.index.tolist()}")
print(f"  Dask new index:   {reset_ddf.index.tolist()}")
print(f"  Match: {reset_pdf.index.tolist() == reset_ddf.index.tolist()}")
print(f"  Pandas 'index' column: {reset_pdf['index'].tolist()}")
print(f"  Dask 'index' column:   {reset_ddf['index'].tolist()}")