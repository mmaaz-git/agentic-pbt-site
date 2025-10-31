import pandas as pd
import io

csv_data = "a,b,c\n0,0.0,0\n0,0.0,:\n"

full_read = pd.read_csv(io.StringIO(csv_data))
chunked_read = pd.concat([chunk for chunk in pd.read_csv(io.StringIO(csv_data), chunksize=1)], ignore_index=True)

print("Full read column 'c':", full_read['c'].values)
print("Chunked read column 'c':", chunked_read['c'].values)
print("Full read column 'c' dtype:", full_read['c'].dtype)
print("Chunked read column 'c' dtype:", chunked_read['c'].dtype)
print("Equal?", (full_read['c'].values == chunked_read['c'].values).all())

print("\nDetailed comparison:")
for i in range(len(full_read)):
    full_val = full_read['c'].iloc[i]
    chunk_val = chunked_read['c'].iloc[i]
    print(f"Row {i}: full={repr(full_val)} (type={type(full_val).__name__}), chunked={repr(chunk_val)} (type={type(chunk_val).__name__})")