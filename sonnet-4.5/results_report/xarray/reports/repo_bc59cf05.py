from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create a CombineKwargDefault object with different old/new values
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

# Set global option to False and add object to a set
OPTIONS["use_new_combine_kwarg_defaults"] = False
print(f"Initial OPTIONS['use_new_combine_kwarg_defaults'] = {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value = {obj._value}")
hash_before = hash(obj)
print(f"Hash before: {hash_before}")

# Add the object to a set
s = {obj}
print(f"Object added to set: {obj in s}")

# Change the global option
OPTIONS["use_new_combine_kwarg_defaults"] = True
print(f"\nChanged OPTIONS['use_new_combine_kwarg_defaults'] = {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value = {obj._value}")
hash_after = hash(obj)
print(f"Hash after: {hash_after}")

# Check if the object is still in the set
print(f"Object still in set: {obj in s}")

# Verify the hashes are different
print(f"\nHashes are equal: {hash_before == hash_after}")

# Demonstrate the issue - object can't be found in set after hash change
print(f"\nAssertion check: obj in s")
assert obj in s, "Object not found in set after hash changed!"