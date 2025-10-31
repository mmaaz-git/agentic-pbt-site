import pandas as pd
from pandas.core.interchange.dataframe import PandasDataFrameXchg
from pandas.core.interchange.column import PandasColumn

# Test case 1: DataFrame with 11 rows split into 5 chunks
print("Test 1: DataFrame with 11 rows split into 5 chunks")
print("-" * 50)
df = pd.DataFrame({'A': range(11)})
interchange_obj = PandasDataFrameXchg(df)
chunks = list(interchange_obj.get_chunks(5))

print(f"DataFrame size: {len(df)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks]}")
print(f"Total rows in all chunks: {sum(chunk.num_rows() for chunk in chunks)}")
for i, chunk in enumerate(chunks):
    if chunk.num_rows() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 2: Minimal failing case - size=1, n_chunks=2
print("\nTest 2: Minimal failing case - size=1, n_chunks=2")
print("-" * 50)
df_small = pd.DataFrame({'A': [0]})
interchange_obj_small = PandasDataFrameXchg(df_small)
chunks_small = list(interchange_obj_small.get_chunks(2))

print(f"DataFrame size: {len(df_small)} rows")
print(f"Number of chunks requested: 2")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks_small]}")
for i, chunk in enumerate(chunks_small):
    if chunk.num_rows() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 3: Column version of the same bug
print("\nTest 3: Column version - 11 rows split into 5 chunks")
print("-" * 50)
series = pd.Series(range(11))
column = PandasColumn(series)
col_chunks = list(column.get_chunks(5))

print(f"Series size: {len(series)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.size() for chunk in col_chunks]}")
for i, chunk in enumerate(col_chunks):
    if chunk.size() == 0:
        print(f"WARNING: Chunk {i} is EMPTY!")

# Test case 4: Edge case - more chunks than rows
print("\nTest 4: Edge case - 3 rows split into 5 chunks")
print("-" * 50)
df_edge = pd.DataFrame({'A': [1, 2, 3]})
interchange_obj_edge = PandasDataFrameXchg(df_edge)
chunks_edge = list(interchange_obj_edge.get_chunks(5))

print(f"DataFrame size: {len(df_edge)} rows")
print(f"Number of chunks requested: 5")
print(f"Chunk sizes: {[chunk.num_rows() for chunk in chunks_edge]}")
empty_chunks = sum(1 for chunk in chunks_edge if chunk.num_rows() == 0)
print(f"Number of empty chunks: {empty_chunks}")