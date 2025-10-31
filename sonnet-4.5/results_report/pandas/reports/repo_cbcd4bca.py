import pandas as pd
import pyarrow as pa

# Create a Series with lists of different lengths
s = pd.Series(
    [[1, 2, 3], [4]],
    dtype=pd.ArrowDtype(pa.list_(pa.int64()))
)

# Try accessing index 0 (this should work)
print("Accessing index 0:")
print(s.list[0])
print()

# Try accessing index 1 (this should fail for the second list)
print("Accessing index 1:")
try:
    result = s.list[1]
    print(result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")