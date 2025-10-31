import Cython.Tempita as tempita

b = tempita.bunch(x=1, y=2, z=3)

print("Testing tempita.bunch(x=1, y=2, z=3)")
print(f"b.x = {b.x}")
print(f"b.y = {b.y}")
print(f"b.z = {b.z}")

print(f"\nhasattr(b, 'x') = {hasattr(b, 'x')}")
print(f"hasattr(b, 'y') = {hasattr(b, 'y')}")
print(f"hasattr(b, 'z') = {hasattr(b, 'z')}")

print(f"\n'x' in dir(b) = {'x' in dir(b)}")
print(f"'y' in dir(b) = {'y' in dir(b)}")
print(f"'z' in dir(b) = {'z' in dir(b)}")

print(f"\ndir(b) = {dir(b)}")

# Test the assertions from the bug report
try:
    assert b.x == 1
    print("\n✓ b.x == 1")
except AssertionError:
    print("\n✗ b.x == 1 failed")

try:
    assert hasattr(b, 'x')
    print("✓ hasattr(b, 'x')")
except AssertionError:
    print("✗ hasattr(b, 'x') failed")

try:
    assert 'x' not in dir(b)
    print("✓ 'x' not in dir(b) - confirming the bug")
except AssertionError:
    print("✗ 'x' not in dir(b) failed - bug not present")