import pandas as pd
import pyarrow as pa

s = pd.Series(
    [[1, 2, 3], [4]],
    dtype=pd.ArrowDtype(pa.list_(pa.int64()))
)

print("Series contents:")
print(s)
print()

print("s.list[0] (accessing first element of each list):")
print(s.list[0])
print()

print("Now trying s.list[1] (accessing second element of each list):")
try:
    result = s.list[1]
    print(result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")