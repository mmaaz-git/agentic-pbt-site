from pandas.core.indexes.frozen import FrozenList

fl1 = FrozenList([1, 2, 3, 4, 5])
fl2 = FrozenList([3, 4, 5, 6])

result = fl1.difference(fl2)

print(f"fl1: {fl1}")
print(f"fl2: {fl2}")
print(f"fl1.difference(fl2): {result}")
print()
print("Expected (based on docstring): difference = [1, 2]")
print("Actual: difference = {result}")