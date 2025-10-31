from Cython.Utility.Dataclasses import field

f = field(kw_only=True, default=42)
print("repr output:")
print(repr(f))
print("\nChecking attribute access:")
print(f"f.kw_only = {f.kw_only}")

# Also test that kwonly is not an attribute
try:
    print(f"f.kwonly = {f.kwonly}")
except AttributeError as e:
    print(f"AttributeError when accessing f.kwonly: {e}")