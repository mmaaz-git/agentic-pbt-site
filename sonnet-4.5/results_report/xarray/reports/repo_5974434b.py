from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create a CombineKwargDefault object with different old and new values
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

# Store the object as a dictionary key
d = {obj: "stored_value"}

# Initial retrieval - this should work
print(f"Initial OPTIONS['use_new_combine_kwarg_defaults']: {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"Initial hash of object: {hash(obj)}")
print(f"Initial value from dict: {d[obj]}")
print()

# Change the global OPTIONS setting
OPTIONS["use_new_combine_kwarg_defaults"] = True
print(f"Changed OPTIONS['use_new_combine_kwarg_defaults'] to: {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"New hash of object: {hash(obj)}")

# Try to retrieve the value again - this will fail
try:
    print(f"Value from dict after OPTIONS change: {d[obj]}")
except KeyError as e:
    print(f"KeyError: Object lost! Hash changed, so dict lookup fails.")
    print(f"The key {obj} is no longer found in the dictionary.")