print("Python str.index behavior:")
print("=" * 60)

# Single string case
s = "hello world"
print(f"String: '{s}'")
print(f"s.index('world'): {s.index('world')}")
print(f"s.find('world'): {s.find('world')}")

try:
    print(f"s.index('xyz'): ", end="")
    print(s.index('xyz'))
except ValueError as e:
    print(f"raises ValueError: {e}")

print(f"s.find('xyz'): {s.find('xyz')}")

print()
print("Documentation for str.index:")
print("-" * 40)
print(str.index.__doc__)

print()
print("Documentation for str.find:")
print("-" * 40)
print(str.find.__doc__)