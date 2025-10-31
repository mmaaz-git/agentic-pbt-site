#!/usr/bin/env python3
"""Test script to reproduce the CombineKwargDefault hash mutability bug"""

from hypothesis import given, strategies as st, assume
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# First test: Property-based test
print("=" * 60)
print("Running property-based test:")
print("=" * 60)

@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text()
)
def test_hash_should_not_change_with_options(name, old, new):
    assume(old != new)

    obj = CombineKwargDefault(name=name, old=old, new=new)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed for name='{name}', old='{old}', new='{new}': {hash1} -> {hash2}"

# Run the property test with specific failing example
try:
    test_failing_case = test_hash_should_not_change_with_options
    test_failing_case(name='0', old='', new='0')
    print("Test unexpectedly passed!")
except AssertionError as e:
    print(f"Property test failed as expected: {e}")

print("\n" + "=" * 60)
print("Running specific reproduction example:")
print("=" * 60)

# Second test: Manual reproduction
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    s = {obj}
    assert obj in s
    print(f"With use_new_combine_kwarg_defaults=False:")
    print(f"  Hash: {hash1}")
    print(f"  Object._value: {obj._value}")
    print(f"  Object in set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"\nWith use_new_combine_kwarg_defaults=True:")
    print(f"  Hash: {hash2}")
    print(f"  Object._value: {obj._value}")
    print(f"  Object in set: {obj in s}")
    print(f"\nHash changed: {hash1} -> {hash2}")
    print(f"Same hash? {hash1 == hash2}")

print("\n" + "=" * 60)
print("Testing dict/set behavior:")
print("=" * 60)

# Test with dictionary
obj2 = CombineKwargDefault(name="dict_test", old="old", new="new")

with set_options(use_new_combine_kwarg_defaults=False):
    d = {obj2: "value_in_dict"}
    print(f"Added to dict with hash: {hash(obj2)}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"New hash after option change: {hash(obj2)}")
    try:
        retrieved = d[obj2]
        print(f"Retrieved from dict: {retrieved}")
    except KeyError:
        print("KeyError: Object cannot be found in dict after hash changed!")

# Additional test: verify equality still works
with set_options(use_new_combine_kwarg_defaults=False):
    obj3_a = CombineKwargDefault(name="eq_test", old="a", new="b")

with set_options(use_new_combine_kwarg_defaults=True):
    obj3_b = CombineKwargDefault(name="eq_test", old="a", new="b")

print(f"\nEquality test:")
print(f"obj3_a == obj3_b: {obj3_a == obj3_b}")
print(f"hash(obj3_a) == hash(obj3_b): {hash(obj3_a) == hash(obj3_b)}")