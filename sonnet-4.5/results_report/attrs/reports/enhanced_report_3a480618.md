# Bug Report: attr.has() Violates Documented TypeError Contract

**Target**: `attr.has`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `attr.has()` function violates its documented API contract by accepting non-class inputs without raising `TypeError`, contradicting its explicit documentation which promises to raise `TypeError` when the input is not a class.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import attr
import inspect

@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
    st.none(),
    st.lists(st.integers()),
    st.dictionaries(st.text(), st.integers()),
))
def test_has_should_raise_for_non_classes(non_class_value):
    """attr.has() should raise TypeError for non-class inputs per its documentation."""
    assume(not inspect.isclass(non_class_value))

    try:
        result = attr.has(non_class_value)
        raise AssertionError(
            f"attr.has({non_class_value!r}) returned {result} instead of raising TypeError"
        )
    except TypeError:
        pass

if __name__ == "__main__":
    test_has_should_raise_for_non_classes()
```

<details>

<summary>
**Failing input**: `None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 26, in <module>
    test_has_should_raise_for_non_classes()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_has_should_raise_for_non_classes
    st.integers(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 19, in test_has_should_raise_for_non_classes
    raise AssertionError(
        f"attr.has({non_class_value!r}) returned {result} instead of raising TypeError"
    )
AssertionError: attr.has(None) returned False instead of raising TypeError
Falsifying example: test_has_should_raise_for_non_classes(
    non_class_value=None,
)
```
</details>

## Reproducing the Bug

```python
import attr

# Test case 1: Integer input
result = attr.has(42)
print(f"attr.has(42) = {result}")

# Test case 2: String input
result = attr.has("not a class")
print(f"attr.has('not a class') = {result}")

# Test case 3: None input
result = attr.has(None)
print(f"attr.has(None) = {result}")

# Test case 4: List input
result = attr.has([1, 2, 3])
print(f"attr.has([1, 2, 3]) = {result}")

# Test case 5: Dictionary input
result = attr.has({"key": "value"})
print(f"attr.has({{'key': 'value'}}) = {result}")

# Test case 6: Float input
result = attr.has(3.14)
print(f"attr.has(3.14) = {result}")

print("\nAll of the above should have raised TypeError according to documentation.")
print("Instead, they all returned False.")
```

<details>

<summary>
Returns False for all non-class inputs instead of raising TypeError
</summary>
```
attr.has(42) = False
attr.has('not a class') = False
attr.has(None) = False
attr.has([1, 2, 3]) = False
attr.has({'key': 'value'}) = False
attr.has(3.14) = False

All of the above should have raised TypeError according to documentation.
Instead, they all returned False.
```
</details>

## Why This Is A Bug

The `attr.has()` function's docstring at `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_funcs.py:326-338` explicitly states:

```python
def has(cls):
    """
    Check whether *cls* is a class with *attrs* attributes.

    Args:
        cls (type): Class to introspect.

    Raises:
        TypeError: If *cls* is not a class.

    Returns:
        bool:
    """
```

The documentation clearly promises to raise `TypeError` if the input is not a class. However, the actual implementation (lines 339-351) does not validate that `cls` is actually a class. Instead, it directly calls `getattr(cls, "__attrs_attrs__", None)` which succeeds on any object type without raising an error.

This creates multiple issues:

1. **Contract Violation**: The function doesn't behave as documented, breaking the API contract that users depend on.

2. **Silent Failures**: Users cannot rely on `TypeError` to catch programming errors where they accidentally pass wrong types. The function silently accepts invalid inputs and returns `False` instead of failing fast.

3. **API Inconsistency**: Other similar functions in the same module DO validate their inputs:
   - `fields()` at line 1861-1863 checks `not isinstance(cls, type)` and raises `TypeError` with "Passed object must be a class."
   - `fields_dict()` at line 1901-1903 performs the same validation

4. **Misleading Parameter Name**: The parameter is named `cls` which conventionally indicates a class is expected, not arbitrary objects.

5. **Type Annotation Mismatch**: The documentation specifies `cls (type)` which clearly indicates only class types should be accepted.

## Relevant Context

The attrs library maintains consistency across its API for introspection functions. Both `fields()` and `fields_dict()` validate their inputs and raise `TypeError` for non-class arguments. The `has()` function appears to be an outlier that was likely overlooked during implementation.

The current permissive behavior might seem convenient, but it violates the principle of explicit error handling and makes debugging harder when users accidentally pass wrong types. The function's purpose is to check if a class has attrs attributes - passing non-classes is clearly a programming error that should be caught early.

Documentation reference: The docstring format and promises are consistent across the module, suggesting this is an implementation oversight rather than intentional design.

## Proposed Fix

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -336,6 +336,10 @@ def has(cls):
     Returns:
         bool:
     """
+    if not isinstance(cls, type):
+        msg = "Passed object must be a class."
+        raise TypeError(msg)
+
     attrs = getattr(cls, "__attrs_attrs__", None)
     if attrs is not None:
         return True
```

This fix aligns `has()` with the behavior of `fields()` and `fields_dict()`, using the same validation check and error message for consistency across the API.