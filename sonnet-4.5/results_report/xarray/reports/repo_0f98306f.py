import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create an instance of CombineKwargDefault
obj = CombineKwargDefault(name='test', old='old_val', new='new_val')

# Save original setting
original_setting = OPTIONS["use_new_combine_kwarg_defaults"]

# Get hash with original setting
hash1 = hash(obj)
print(f"Hash with OPTIONS['use_new_combine_kwarg_defaults']={original_setting}: {hash1}")

# Change the global OPTIONS setting
OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting

# Get hash with changed setting
hash2 = hash(obj)
print(f"Hash with OPTIONS['use_new_combine_kwarg_defaults']={not original_setting}: {hash2}")

# Check if hash changed
if hash1 != hash2:
    print("\n❌ BUG: Hash changed when OPTIONS setting changed!")
    print(f"   This violates Python's hash contract.")

    # Demonstrate the practical impact: object gets lost in dictionary
    print("\nDemonstrating dictionary lookup failure:")

    # Reset to original setting
    OPTIONS["use_new_combine_kwarg_defaults"] = original_setting

    # Create dictionary with object as key
    test_dict = {obj: "value"}
    print(f"  Created dict with obj as key: {test_dict}")

    # Change OPTIONS again
    OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting

    # Try to retrieve value - this will fail
    try:
        value = test_dict[obj]
        print(f"  Retrieved value: {value}")
    except KeyError:
        print(f"  ❌ KeyError: Object not found in dictionary after OPTIONS change!")
        print(f"     The object is 'lost' because its hash changed")
else:
    print("✓ Hash remained constant (expected behavior)")

# Reset OPTIONS to original value
OPTIONS["use_new_combine_kwarg_defaults"] = original_setting