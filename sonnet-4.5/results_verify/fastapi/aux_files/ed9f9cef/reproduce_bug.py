import attr

@attr.define(hash=True)
class TestClass:
    x: int
    weird: str = attr.field(eq=False, hash=True)

instance1 = TestClass(x=1, weird="text1")
instance2 = TestClass(x=1, weird="text2")

print(f"instance1 == instance2: {instance1 == instance2}")
print(f"hash(instance1): {hash(instance1)}")
print(f"hash(instance2): {hash(instance2)}")

# Check if hash/equality contract is violated
if instance1 == instance2:
    if hash(instance1) != hash(instance2):
        print("\nBUG CONFIRMED: Equal objects have different hashes!")
        print("This violates Python's hash/equality contract.")
    else:
        print("\nNo bug: Equal objects have the same hash.")
else:
    print("\nObjects are not equal, so different hashes are allowed.")