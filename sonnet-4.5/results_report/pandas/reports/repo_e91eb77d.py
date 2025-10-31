import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg

# Test case 1: 1 row, 2 chunks requested
print("Test case 1: 1 row, 2 chunks requested")
df = pd.DataFrame({'a': [1]})
xchg_df = PandasDataFrameXchg(df)

chunks = list(xchg_df.get_chunks(n_chunks=2))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.num_rows()} rows")

print("\n" + "="*50 + "\n")

# Test case 2: 5 rows, 10 chunks requested
print("Test case 2: 5 rows, 10 chunks requested")
df2 = pd.DataFrame({'a': range(5)})
xchg_df2 = PandasDataFrameXchg(df2)

chunks2 = list(xchg_df2.get_chunks(n_chunks=10))

for i, chunk in enumerate(chunks2):
    print(f"Chunk {i}: {chunk.num_rows()} rows")