from Cython.Utility.Dataclasses import field, MISSING

f = field(default=MISSING, kw_only=True)

print("Field attributes:")
print(f"  f.kw_only = {f.kw_only}")

print("\nField repr:")
print(f"  {repr(f)}")

print("\nChecking repr content:")
if 'kw_only=True' in repr(f):
    print("  ✓ repr contains 'kw_only=True'")
else:
    print("  ✗ repr does NOT contain 'kw_only=True'")

if 'kwonly=True' in repr(f):
    print("  ✓ repr contains 'kwonly=True'")
else:
    print("  ✗ repr does NOT contain 'kwonly=True'")