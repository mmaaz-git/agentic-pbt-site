from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

# Test 1: Hash changes with global OPTIONS
print("Test 1: Hash mutation with global OPTIONS")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=False: {hash1}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=True: {hash2}")

print(f"Hash changed: {hash1 != hash2}")
print()

# Test 2: Set membership breaks
print("Test 2: Set membership failure")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    s = {obj}
    print(f"Object added to set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after option change: {obj in s}")
print()

# Test 3: Dictionary key lookup fails
print("Test 3: Dictionary key lookup failure")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    d = {obj: "value"}
    print(f"Object added as dict key: {obj in d}")

with set_options(use_new_combine_kwarg_defaults=True):
    try:
        value = d[obj]
        print(f"Successfully retrieved value: {value}")
    except KeyError:
        print("KeyError: Object cannot be found as dictionary key after option change")
print()

# Test 4: Show the failing case from the report
print("Test 4: Specific failing case from report")
print("-" * 50)

obj2 = CombineKwargDefault(name='0', old='', new='0')

with set_options(use_new_combine_kwarg_defaults=False):
    hash3 = hash(obj2)
    print(f"Hash with False (old=''): {hash3}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash4 = hash(obj2)
    print(f"Hash with True (new='0'): {hash4}")

print(f"Hash changed: {hash3 != hash4}")