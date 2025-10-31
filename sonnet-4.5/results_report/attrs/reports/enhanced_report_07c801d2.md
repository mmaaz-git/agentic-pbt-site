# Bug Report: attrs Hash/Equality Contract Violation with eq=False and hash=True

**Target**: `attrs.field` with `eq=False` and `hash=True`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The attrs library allows fields to have `eq=False` (excluded from equality) and `hash=True` (included in hash), which violates Python's fundamental requirement that equal objects must have equal hashes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import attrs

@given(st.integers(), st.integers(), st.integers())
def test_hash_equality_contract_with_eq_false_hash_true(shared, val1, val2):
    """Objects that are equal MUST have equal hashes (Python requirement)"""
    @attrs.define(hash=True)
    class Data:
        shared: int
        excluded: int = attrs.field(eq=False, hash=True)

    obj1 = Data(shared, val1)
    obj2 = Data(shared, val2)

    assume(val1 != val2)

    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)

if __name__ == "__main__":
    test_hash_equality_contract_with_eq_false_hash_true()
```

<details>

<summary>
**Failing input**: `Data(0, 0)` and `Data(0, 1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 21, in <module>
    test_hash_equality_contract_with_eq_false_hash_true()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 5, in test_hash_equality_contract_with_eq_false_hash_true
    def test_hash_equality_contract_with_eq_false_hash_true(shared, val1, val2):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 18, in test_hash_equality_contract_with_eq_false_hash_true
    assert hash(obj1) == hash(obj2)
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_hash_equality_contract_with_eq_false_hash_true(
    # The test always failed when commented parts were varied together.
    shared=0,  # or any other generated value
    val1=0,  # or any other generated value
    val2=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import attrs

@attrs.define(hash=True)
class Data:
    shared: int
    excluded: int = attrs.field(eq=False, hash=True)

obj1 = Data(0, 1)
obj2 = Data(0, 2)

print(f"obj1 == obj2: {obj1 == obj2}")
print(f"hash(obj1): {hash(obj1)}")
print(f"hash(obj2): {hash(obj2)}")
print(f"Contract violated: {obj1 == obj2 and hash(obj1) != hash(obj2)}")

# Demonstrate the problem with dictionaries
d = {obj1: "first"}
d[obj2] = "second"

print(f"\nDictionary behavior:")
print(f"Number of items in dict: {len(d)}")
print(f"obj1 == obj2: {obj1 == obj2}")
print("Two equal objects are distinct dictionary keys!")

# Demonstrate the problem with sets
s = {obj1}
s.add(obj2)
print(f"\nSet behavior:")
print(f"Number of items in set: {len(s)}")
print(f"obj1 == obj2: {obj1 == obj2}")
print("Two equal objects are distinct set members!")
```

<details>

<summary>
Output showing contract violation and incorrect dictionary/set behavior
</summary>
```
obj1 == obj2: True
hash(obj1): 1778154771209699592
hash(obj2): -3114919257762996774
Contract violated: True

Dictionary behavior:
Number of items in dict: 2
obj1 == obj2: True
Two equal objects are distinct dictionary keys!

Set behavior:
Number of items in set: 2
obj1 == obj2: True
Two equal objects are distinct set members!
```
</details>

## Why This Is A Bug

This violates Python's fundamental hash/equality contract as documented in the Python data model:

> "The only required property is that objects which compare equal have the same hash value"

When a field is marked with `eq=False`, it is excluded from equality comparisons. However, when the same field has `hash=True`, it is included in hash calculations. This creates a situation where two objects can be equal (because they only differ in fields excluded from equality) but have different hash values (because those excluded fields are still used in the hash).

This violation causes critical failures in Python's hash-based collections:

1. **Dictionaries break**: Equal objects with different hashes can exist as separate keys, violating the expectation that `d[obj1]` and `d[obj2]` refer to the same entry when `obj1 == obj2`.

2. **Sets break**: Equal objects are treated as distinct members, violating the mathematical definition of a set.

3. **Cache lookups fail**: Any code using objects as cache keys will malfunction.

The attrs documentation itself acknowledges this is incorrect behavior, stating in the field() docstring: "If None (default), mirror *eq*'s value. This is the correct behavior according the Python spec. Setting this value to anything else than None is *discouraged*."

## Relevant Context

The issue occurs in `/home/npc/miniconda/lib/python3.13/site-packages/attr/_next_gen.py:508-510` where the documentation states:

> "Include this attribute in the generated `__hash__` method. If None (default), mirror *eq*'s value. This is the correct behavior according the Python spec. Setting this value to anything else than None is *discouraged*."

Despite this acknowledgment that deviating from mirroring eq is incorrect, the library still allows the invalid combination without any validation or warning.

The attrs library already validates other invalid field combinations. For example, in the attrib function, it validates that `eq=False` and `order=True` is invalid because ordering requires equality.

Documentation reference: https://www.attrs.org/en/stable/api.html#attrs.field

## Proposed Fix

Add validation to prevent the invalid `eq=False, hash=True` combination by raising a ValueError, similar to existing validation for other invalid combinations:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -160,6 +160,11 @@ def attrib(
     eq, eq_key, order, order_key = _determine_attrib_eq_order(
         cmp, eq, order, True
     )
+
+    # Validate hash/eq compatibility
+    if eq is False and hash is True:
+        msg = "Cannot set hash=True when eq=False. This violates Python's hash/equality contract: equal objects must have equal hashes."
+        raise ValueError(msg)

     if hash is not None and hash is not True and hash is not False:
         msg = "Invalid value for hash.  Must be True, False, or None."
```

A similar check should be added to the `field()` function in `_next_gen.py`. Alternatively, the library could automatically set `hash=False` when `eq=False` is specified, with a warning to inform the user.