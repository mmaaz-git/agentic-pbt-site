# Bug Report: xarray.core.utils.OrderedSet.discard() Violates MutableSet Contract by Raising KeyError

**Target**: `xarray.core.utils.OrderedSet.discard()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `OrderedSet.discard()` method raises `KeyError` when attempting to discard a non-existent element, violating the Python `MutableSet` abstract base class contract which requires `discard()` to silently do nothing when the element is not present.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.utils import OrderedSet

@given(st.lists(st.integers()), st.integers())
def test_orderedset_discard_never_raises(initial_values, value_to_discard):
    """
    Property: discard() should never raise an error, whether the element
    exists or not. This is the core contract of MutableSet.discard().
    """
    os = OrderedSet(initial_values)
    os.discard(value_to_discard)
    # If we reach here without raising, the test passes

# Run the test
if __name__ == "__main__":
    test_orderedset_discard_never_raises()
```

<details>

<summary>
**Failing input**: `OrderedSet([])`, attempting to `discard(0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 16, in <module>
    test_orderedset_discard_never_raises()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 5, in test_orderedset_discard_never_raises
    def test_orderedset_discard_never_raises(initial_values, value_to_discard):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/47/hypo.py", line 11, in test_orderedset_discard_never_raises
    os.discard(value_to_discard)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/utils.py", line 600, in discard
    del self._d[value]
        ~~~~~~~^^^^^^^
KeyError: 0
Falsifying example: test_orderedset_discard_never_raises(
    initial_values=[],
    value_to_discard=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from xarray.core.utils import OrderedSet

# Create an OrderedSet with some initial values
os = OrderedSet([1, 2, 3])
print(f"Initial OrderedSet: {os}")

# Try to discard an element that doesn't exist
# According to the MutableSet contract, this should NOT raise an error
print("\nAttempting to discard(999) - an element not in the set...")
try:
    os.discard(999)
    print("Success: discard(999) completed without error")
except KeyError as e:
    print(f"ERROR: KeyError raised: {e}")

# For comparison, test with built-in set
print("\n--- Comparison with built-in set ---")
s = {1, 2, 3}
print(f"Initial set: {s}")
print("Attempting to discard(999) - an element not in the set...")
try:
    s.discard(999)
    print("Success: discard(999) completed without error")
except KeyError as e:
    print(f"ERROR: KeyError raised: {e}")

# Test discarding an existing element (should work for both)
print("\n--- Testing discard of existing element ---")
os2 = OrderedSet([1, 2, 3])
print(f"OrderedSet before discard(2): {os2}")
os2.discard(2)
print(f"OrderedSet after discard(2): {os2}")
```

<details>

<summary>
OrderedSet.discard() raises KeyError for non-existent element while built-in set.discard() succeeds silently
</summary>
```
Initial OrderedSet: OrderedSet([1, 2, 3])

Attempting to discard(999) - an element not in the set...
ERROR: KeyError raised: 999

--- Comparison with built-in set ---
Initial set: {1, 2, 3}
Attempting to discard(999) - an element not in the set...
Success: discard(999) completed without error

--- Testing discard of existing element ---
OrderedSet before discard(2): OrderedSet([1, 2, 3])
OrderedSet after discard(2): OrderedSet([1, 3])
```
</details>

## Why This Is A Bug

This violates the fundamental contract of Python's `collections.abc.MutableSet` abstract base class in multiple ways:

1. **MutableSet Contract Violation**: The `MutableSet` ABC explicitly documents that `discard(value)` should "Remove an element. Do not raise an exception if absent." The current implementation directly violates this requirement by raising `KeyError` when the element is not present.

2. **Liskov Substitution Principle Violation**: Since `OrderedSet` inherits from `MutableSet[T]`, it should be usable as a drop-in replacement for any MutableSet. Code that expects a MutableSet and calls `discard()` on non-existent elements will unexpectedly crash when given an OrderedSet instance.

3. **API Inconsistency**: The OrderedSet class documentation at line 572-573 states "The API matches the builtin set", but the built-in `set.discard()` does not raise an exception for missing elements. This creates a false expectation for users.

4. **Semantic Difference from remove()**: The distinction between `discard()` and `remove()` in the set API is that `remove()` raises KeyError for missing elements while `discard()` does not. The current implementation makes both methods behave identically, eliminating this important semantic difference.

## Relevant Context

The bug is located in `/home/npc/miniconda/lib/python3.13/site-packages/xarray/core/utils.py` at lines 599-600:

```python
def discard(self, value: T) -> None:
    del self._d[value]
```

The issue is that `del self._d[value]` unconditionally attempts to delete the key from the internal dictionary, which raises `KeyError` if the key doesn't exist. This is the wrong approach for implementing `discard()`.

For reference, Python's built-in set documentation states:
- `set.discard(elem)`: "Remove element elem from the set if it is present."
- `set.remove(elem)`: "Remove element elem from the set. Raises KeyError if elem is not contained in the set."

The MutableSet ABC in `collections.abc` enforces this same contract, with `remove()` being implemented in terms of membership checking followed by `discard()`.

## Proposed Fix

```diff
--- a/xarray/core/utils.py
+++ b/xarray/core/utils.py
@@ -597,7 +597,7 @@ class OrderedSet(MutableSet[T]):
         self._d[value] = None

     def discard(self, value: T) -> None:
-        del self._d[value]
+        self._d.pop(value, None)

     # Additional methods
```