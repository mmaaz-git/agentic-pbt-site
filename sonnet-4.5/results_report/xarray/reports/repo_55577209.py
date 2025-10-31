from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object with different old and new values
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

# Get initial hash
hash1 = hash(obj)
print(f"Initial hash: {hash1}")

# Change OPTIONS and get hash again
with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash after set_options(use_new_combine_kwarg_defaults=True): {hash2}")

# Back to original OPTIONS
with set_options(use_new_combine_kwarg_defaults=False):
    hash3 = hash(obj)
    print(f"Hash after set_options(use_new_combine_kwarg_defaults=False): {hash3}")

print(f"\nAll hashes equal: {hash1 == hash2 == hash3}")

# Demonstrate the practical issue: object lost in set
print("\n--- Demonstrating set/dict issue ---")
s = {obj}
print(f"Object added to set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after OPTIONS change: {obj in s}")

# Demonstrate the issue with dictionaries too
print("\n--- Demonstrating dictionary issue ---")
d = {obj: "value"}
print(f"Object used as dict key: {obj in d}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object as dict key after OPTIONS change: {obj in d}")