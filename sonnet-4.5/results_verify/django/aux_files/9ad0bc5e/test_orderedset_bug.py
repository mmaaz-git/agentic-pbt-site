#!/usr/bin/env python3
"""Test the reported OrderedSet equality bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.datastructures import OrderedSet
from hypothesis import given, settings, strategies as st

print("=" * 60)
print("Testing OrderedSet Equality Bug Report")
print("=" * 60)

# Test 1: Simple equality test from bug report
print("\nTest 1: Simple equality test")
os1 = OrderedSet([1, 2, 3])
os2 = OrderedSet([1, 2, 3])
print(f"OrderedSet([1, 2, 3]) created as os1")
print(f"OrderedSet([1, 2, 3]) created as os2")
print(f"os1 == os2: {os1 == os2}")
print(f"os1 is os2: {os1 is os2}")

# Test 2: Empty OrderedSet
print("\nTest 2: Empty OrderedSet equality")
empty1 = OrderedSet()
empty2 = OrderedSet()
print(f"empty1 == empty2: {empty1 == empty2}")

# Test 3: Same OrderedSet compared to itself
print("\nTest 3: Self equality")
print(f"os1 == os1: {os1 == os1}")

# Test 4: Different order (OrderedSet should preserve order)
print("\nTest 4: Different order test")
os3 = OrderedSet([3, 2, 1])
print(f"OrderedSet([3, 2, 1]) created as os3")
print(f"os1 == os3: {os1 == os3}")
print(f"Contents of os1: {list(os1)}")
print(f"Contents of os3: {list(os3)}")

# Test 5: Property-based test from bug report
print("\nTest 5: Running property-based test")
try:
    @given(st.lists(st.integers()))
    @settings(max_examples=10)  # Just a few examples for demo
    def test_orderedset_equality_reflexive(items):
        os1 = OrderedSet(items)
        os2 = OrderedSet(items)
        assert os1 == os2, f"OrderedSet({items}) should equal OrderedSet({items})"

    test_orderedset_equality_reflexive()
    print("Property test PASSED (unexpected!)")
except AssertionError as e:
    print(f"Property test FAILED as expected: {e}")

# Test 6: Check other operations
print("\nTest 6: Other OrderedSet operations")
os_test = OrderedSet([1, 2, 3, 4])
print(f"OrderedSet: {os_test}")
print(f"Length: {len(os_test)}")
print(f"Contains 2: {2 in os_test}")
print(f"Contains 5: {5 in os_test}")
print(f"Iteration: {list(os_test)}")
print(f"Reversed: {list(reversed(os_test))}")

# Test 7: Check if __hash__ exists
print("\nTest 7: Check __hash__ implementation")
try:
    hash(os1)
    print("OrderedSet is hashable")
except TypeError as e:
    print(f"OrderedSet is NOT hashable: {e}")

# Test 8: Check what equality currently uses
print("\nTest 8: Identity vs Equality")
os_copy1 = OrderedSet([1, 2, 3])
os_copy2 = OrderedSet([1, 2, 3])
print(f"id(os_copy1): {id(os_copy1)}")
print(f"id(os_copy2): {id(os_copy2)}")
print(f"os_copy1 == os_copy2: {os_copy1 == os_copy2}")
print(f"os_copy1 is os_copy2: {os_copy1 is os_copy2}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("OrderedSet does NOT implement __eq__, falling back to identity")
print("This means two OrderedSets with same content are not equal")
print("=" * 60)