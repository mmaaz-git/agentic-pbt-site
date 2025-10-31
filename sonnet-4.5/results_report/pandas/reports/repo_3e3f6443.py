import pandas as pd

df = pd.DataFrame({'a': [0]})
interchange_df = df.__dataframe__()

chunks = list(interchange_df.get_chunks(n_chunks=2))

print(f"DataFrame has {df.shape[0]} rows")
print(f"Requested {2} chunks")
print(f"Got {len(chunks)} chunks")
print(f"Chunk 0: {chunks[0].num_rows()} rows")
print(f"Chunk 1: {chunks[1].num_rows()} rows")