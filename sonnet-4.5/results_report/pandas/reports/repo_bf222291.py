import pandas as pd
import io

csv_data = "a,b,c\n0,0.0,0\n0,0.0,:\n"

full_read = pd.read_csv(io.StringIO(csv_data))
chunked_read = pd.concat([chunk for chunk in pd.read_csv(io.StringIO(csv_data), chunksize=1)], ignore_index=True)

print("Full read column 'c':", full_read['c'].values)
print("Full read column 'c' dtype:", full_read['c'].dtype)
print()
print("Chunked read column 'c':", chunked_read['c'].values)
print("Chunked read column 'c' dtype:", chunked_read['c'].dtype)
print()
print("Equal?", (full_read['c'].values == chunked_read['c'].values).all())
print()
print("Full DataFrame:")
print(full_read)
print("\nChunked DataFrame:")
print(chunked_read)