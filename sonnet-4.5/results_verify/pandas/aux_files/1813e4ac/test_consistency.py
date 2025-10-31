import pandas as pd
import numpy as np
import pyarrow as pa

# Test 1: Regular list accessor behavior (non-arrow)
print("=" * 50)
print("Regular Python lists in Series:")
s = pd.Series([[1, 2, 3], [4, 5], [6]])
print(s)
print("\nUsing .iloc[0][1] on first element:")
print(s.iloc[0][1])  # Should return 2

print("\nUsing apply to get index 1 from each list:")
print(s.apply(lambda x: x[1] if len(x) > 1 else None))

# Test 2: String accessor behavior
print("\n" + "=" * 50)
print("String accessor behavior:")
s = pd.Series(['abc', 'de', 'f'])
print(s)
print("\nUsing .str[2] (out of bounds for 'de' and 'f'):")
print(s.str[2])

# Test 3: Check if any other Arrow accessors exist
print("\n" + "=" * 50)
print("Checking Arrow string behavior:")
s_arrow = pd.Series(['abc', 'de', 'f'], dtype=pd.ArrowDtype(pa.string()))
print(s_arrow)
print("\nUsing .str[2] on arrow strings:")
print(s_arrow.str[2])

# Test 4: Check regular Series indexing behavior
print("\n" + "=" * 50)
print("Regular Series indexing with out of bounds:")
s = pd.Series([1, 2, 3])
print("Series:", s.tolist())
print("\nUsing .iloc[0]:", s.iloc[0])
try:
    print("Using .iloc[5]:", s.iloc[5])
except Exception as e:
    print(f"Using .iloc[5]: {type(e).__name__}: {e}")

# Test 5: Check DataFrame column indexing behavior
print("\n" + "=" * 50)
print("DataFrame column access with missing columns:")
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(df)
print("\nAccessing column 'A':")
print(df['A'])
try:
    print("\nAccessing non-existent column 'C':")
    print(df['C'])
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")