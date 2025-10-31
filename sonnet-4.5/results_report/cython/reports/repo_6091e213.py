from Cython.Utility.Dataclasses import field

# Create a field with kw_only=True and default=42
f = field(kw_only=True, default=42)

# Print the repr output to show the issue
print("repr(f) output:")
print(repr(f))

# Show that the actual attribute is 'kw_only' not 'kwonly'
print("\nAccessing f.kw_only:")
print(f"f.kw_only = {f.kw_only}")

print("\nTrying to access f.kwonly (will raise AttributeError):")
try:
    print(f.kwonly)
except AttributeError as e:
    print(f"AttributeError: {e}")