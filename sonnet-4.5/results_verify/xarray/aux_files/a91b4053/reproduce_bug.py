from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

hash1 = hash(obj)

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)

print(f"Hash before: {hash1}")
print(f"Hash after changing OPTIONS: {hash2}")
print(f"Hashes equal: {hash1 == hash2}")

s = {obj}
with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after OPTIONS change: {obj in s}")