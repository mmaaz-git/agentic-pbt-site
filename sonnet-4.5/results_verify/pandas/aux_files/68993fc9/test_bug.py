import pandas as pd
from pandas.core.indexes.frozen import FrozenList

fl1 = FrozenList([1, 2, 3])
fl2 = FrozenList([4, 5, 6])

result = fl1.union(fl2)

print(f"fl1: {fl1}")
print(f"fl2: {fl2}")
print(f"fl1.union(fl2): {result}")
print()
print("Expected (based on docstring): difference = [1, 2, 3]")
print("Actual: concatenation = [1, 2, 3, 4, 5, 6]")