import attrs

@attrs.define(hash=True)
class Data:
    shared: int
    excluded: int = attrs.field(eq=False, hash=True)

obj1 = Data(0, 1)
obj2 = Data(0, 2)

print(f"obj1 == obj2: {obj1 == obj2}")
print(f"hash(obj1): {hash(obj1)}")
print(f"hash(obj2): {hash(obj2)}")
print(f"Contract violated: {obj1 == obj2 and hash(obj1) != hash(obj2)}")

# Demonstrate the problem with dictionaries
d = {obj1: "first"}
d[obj2] = "second"

print(f"\nDictionary behavior:")
print(f"Number of items in dict: {len(d)}")
print(f"obj1 == obj2: {obj1 == obj2}")
print("Two equal objects are distinct dictionary keys!")

# Demonstrate the problem with sets
s = {obj1}
s.add(obj2)
print(f"\nSet behavior:")
print(f"Number of items in set: {len(s)}")
print(f"obj1 == obj2: {obj1 == obj2}")
print("Two equal objects are distinct set members!")