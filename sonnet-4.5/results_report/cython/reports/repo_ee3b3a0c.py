import Cython.Tempita as tempita

# Create a bunch object with some attributes
b = tempita.bunch(x=1, y=2, z=3)

# Show that attributes are accessible
print("Accessing attributes:")
print(f"b.x = {b.x}")
print(f"b.y = {b.y}")
print(f"b.z = {b.z}")
print()

# Show that hasattr returns True
print("Using hasattr:")
print(f"hasattr(b, 'x') = {hasattr(b, 'x')}")
print(f"hasattr(b, 'y') = {hasattr(b, 'y')}")
print(f"hasattr(b, 'z') = {hasattr(b, 'z')}")
print()

# Show that these attributes are NOT in dir()
print("Using dir():")
print(f"'x' in dir(b) = {'x' in dir(b)}")
print(f"'y' in dir(b) = {'y' in dir(b)}")
print(f"'z' in dir(b) = {'z' in dir(b)}")
print()

# Show what dir() actually returns (filtered for brevity)
print("Attributes in dir(b) (non-dunder methods):")
non_dunder = [attr for attr in dir(b) if not attr.startswith('__')]
print(non_dunder)