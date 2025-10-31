import Cython.Tempita as tempita

b = tempita.bunch(x=1, y=2)

print(f"b.x = {b.x}")
assert b.x == 1

b.x = 10
print(f"After b.x = 10, b.x = {b.x}")
assert b.x == 10

print("\nTrying delattr(b, 'x')...")
try:
    delattr(b, 'x')
    print("ERROR: delattr succeeded when it should have failed according to the bug report")
except AttributeError as e:
    print(f"delattr raised AttributeError: {e}")

# Let's also test if del b.x works
print("\nTrying del b.x...")
try:
    del b.x
    print("ERROR: del b.x succeeded")
except AttributeError as e:
    print(f"del b.x raised AttributeError: {e}")

# Check that setattr works
print("\nTesting setattr...")
setattr(b, 'z', 100)
print(f"After setattr(b, 'z', 100), b.z = {b.z}")

# Check getattr works
print("\nTesting getattr...")
val = getattr(b, 'y')
print(f"getattr(b, 'y') = {val}")