#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

print("=== Testing Hash Invariant ===")

# First, let's test with a simple example
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

print(f"Initial OPTIONS['use_new_combine_kwarg_defaults']: {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value: {obj._value}")
print(f"Initial hash: {hash(obj)}")

# Store original state
original_option = OPTIONS["use_new_combine_kwarg_defaults"]
original_hash = hash(obj)

# Change the global option
OPTIONS["use_new_combine_kwarg_defaults"] = not original_option
print(f"\nAfter toggling OPTIONS to {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value: {obj._value}")
new_hash = hash(obj)
print(f"New hash: {new_hash}")

print(f"\nHash changed: {original_hash != new_hash}")

# Restore
OPTIONS["use_new_combine_kwarg_defaults"] = original_option

print("\n=== Dictionary Key Test ===")

# Now test the dictionary issue
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

d = {obj: "stored_value"}
print(f"Stored object in dict with key: {obj}")
print(f"Can retrieve value: {d[obj]}")

OPTIONS["use_new_combine_kwarg_defaults"] = not OPTIONS["use_new_combine_kwarg_defaults"]
print(f"\nAfter changing OPTIONS...")

try:
    print(f"Trying to retrieve value: {d[obj]}")
except KeyError:
    print("KeyError! Object lost! Hash changed, so dict lookup fails.")

# Restore
OPTIONS["use_new_combine_kwarg_defaults"] = not OPTIONS["use_new_combine_kwarg_defaults"]

print("\n=== Hypothesis Test ===")

@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text(),
)
@settings(max_examples=100)
def test_hash_remains_constant_during_object_lifetime(name, old, new):
    """
    Property: An object's hash must remain constant during its lifetime.
    """
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_hash = hash(obj)
    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_option
    new_hash = hash(obj)
    OPTIONS["use_new_combine_kwarg_defaults"] = original_option

    assert original_hash == new_hash, (
        f"Hash changed when global OPTIONS changed! "
        f"Before: {original_hash}, After: {new_hash}. "
        f"This violates Python's hash invariant."
    )

try:
    test_hash_remains_constant_during_object_lifetime()
    print("Hypothesis test passed (no issues found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

print("\n=== Testing when old == new ===")
obj_same = CombineKwargDefault(name="test", old="same", new="same")
original_hash_same = hash(obj_same)
OPTIONS["use_new_combine_kwarg_defaults"] = not OPTIONS["use_new_combine_kwarg_defaults"]
new_hash_same = hash(obj_same)
OPTIONS["use_new_combine_kwarg_defaults"] = not OPTIONS["use_new_combine_kwarg_defaults"]
print(f"When old==new, hash changes: {original_hash_same != new_hash_same}")