# Bug Report: attrs validators.gt() Incorrect Docstring

**Target**: `attrs.validators.gt`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validators.gt()` function has incorrect documentation stating it uses `operator.ge` when it actually uses `operator.gt`.

## Property-Based Test

This bug was discovered through code inspection while testing validator properties. The property being tested was documentation accuracy across comparison validators.

```python
from hypothesis import given, strategies as st
from attrs import validators
import operator

@given(st.sampled_from([validators.lt, validators.le, validators.ge, validators.gt]))
def test_validator_docstring_matches_implementation(validator_func):
    validator = validator_func(0)
    docstring = validator_func.__doc__
    actual_op = validator.compare_func

    if 'operator.lt' in docstring:
        assert actual_op == operator.lt
    elif 'operator.le' in docstring:
        assert actual_op == operator.le
    elif 'operator.ge' in docstring:
        assert actual_op == operator.ge
    elif 'operator.gt' in docstring:
        assert actual_op == operator.gt
```

**Failing case**: `validators.gt` - docstring mentions `operator.ge` but uses `operator.gt`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import operator
from attrs import validators

gt_validator = validators.gt(10)

print("Docstring says: 'The validator uses `operator.ge`'")
print(f"Actual implementation: {gt_validator.compare_func}")
print(f"operator.gt: {operator.gt}")
print(f"operator.ge: {operator.ge}")
print(f"Uses operator.gt: {gt_validator.compare_func == operator.gt}")
print(f"Uses operator.ge: {gt_validator.compare_func == operator.ge}")
```

## Why This Is A Bug

The docstring at `attr/validators.py:488` states:

```python
def gt(val):
    """
    ...
    The validator uses `operator.ge` to compare the values.
    ...
    """
    return _NumberValidator(val, ">", operator.gt)
```

The documentation claims `operator.ge` is used, but the implementation clearly uses `operator.gt`. This is inconsistent with the other comparison validators (`lt`, `le`, `ge`) which all have accurate documentation.

The implementation is correct (gt should use operator.gt), but the documentation is wrong.

## Fix

```diff
--- a/attr/validators.py
+++ b/attr/validators.py
@@ -485,7 +485,7 @@ def gt(val):
     A validator that raises `ValueError` if the initializer is called with a
     number smaller or equal to *val*.

-    The validator uses `operator.ge` to compare the values.
+    The validator uses `operator.gt` to compare the values.

     Args:
        val: Exclusive lower bound for values
```