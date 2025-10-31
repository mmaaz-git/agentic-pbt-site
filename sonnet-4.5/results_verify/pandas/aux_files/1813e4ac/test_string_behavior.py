import pandas as pd
import numpy as np

# Test how pandas handles string accessor with different lengths
s = pd.Series(['abc', 'x'])
print("Series with strings of different lengths:")
print(s)
print()

print("Using .str.get(0):")
print(s.str.get(0))
print()

print("Using .str.get(1):")
print(s.str.get(1))
print()

print("Using .str.get(2) (out of bounds for second string):")
print(s.str.get(2))
print()

# Test with .str[index]
print("Using .str[0]:")
print(s.str[0])
print()

print("Using .str[1]:")
print(s.str[1])
print()

print("Using .str[2] (out of bounds for second string):")
print(s.str[2])
print()

# Test if there's a similar pattern with regular Python lists
s_lists = pd.Series([[1, 2, 3], [4]])
print("\nSeries with regular Python lists:")
print(s_lists)
print()

# Try to access elements (this won't work directly)
print("Try accessing element via apply:")
print(s_lists.apply(lambda x: x[0] if len(x) > 0 else None))
print()

print("Try accessing element 1 via apply:")
print(s_lists.apply(lambda x: x[1] if len(x) > 1 else None))
print()

print("Try accessing element 2 via apply:")
print(s_lists.apply(lambda x: x[2] if len(x) > 2 else None))