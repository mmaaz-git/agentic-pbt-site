from Cython.Utility.Dataclasses import field, MISSING

f = field(default=MISSING, kw_only=True)

print("Field attributes:")
print(f"  f.kw_only = {f.kw_only}")

print("\nField repr:")
print(f"  {repr(f)}")