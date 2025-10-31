#!/usr/bin/env python3
"""Test script to reproduce OrderedSet.discard() bug"""

from hypothesis import given, strategies as st
from xarray.core.utils import OrderedSet

# First, test the hypothesis property test
@given(st.lists(st.integers()), st.integers())
def test_orderedset_discard_never_raises(initial_values, value_to_discard):
    """
    Property: discard() should never raise an error, whether the element
    exists or not. This is the core contract of MutableSet.discard().
    """
    os = OrderedSet(initial_values)
    os.discard(value_to_discard)

# Run hypothesis test
print("Testing with hypothesis...")
try:
    test_orderedset_discard_never_raises()
    print("Hypothesis test passed (unexpected)")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

# Direct reproduction
print("\nDirect reproduction test:")
print("Creating OrderedSet([1, 2, 3])...")
os = OrderedSet([1, 2, 3])

print("Attempting to discard 999 (not in set)...")
try:
    os.discard(999)
    print("discard(999) succeeded without error")
except KeyError as e:
    print(f"discard(999) raised KeyError: {e}")
except Exception as e:
    print(f"discard(999) raised unexpected error: {e}")

# Test with built-in set for comparison
print("\nComparison with built-in set:")
s = {1, 2, 3}
print("Creating regular set {1, 2, 3}...")
print("Attempting to discard 999 (not in set)...")
try:
    s.discard(999)
    print("discard(999) succeeded without error (expected behavior)")
except Exception as e:
    print(f"discard(999) raised error: {e}")

# Test discard on existing element
print("\nTest discard on existing element:")
os2 = OrderedSet([1, 2, 3])
print("Creating OrderedSet([1, 2, 3])...")
print("Attempting to discard 2 (in set)...")
try:
    os2.discard(2)
    print(f"discard(2) succeeded. Remaining set: {list(os2)}")
except Exception as e:
    print(f"discard(2) raised error: {e}")