# Bug Report: attr.asdict value_serializer Documentation Incomplete

**Target**: `attr.asdict` (function in `/lib/python3.13/site-packages/attr/_funcs.py`)
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `asdict()` function's `value_serializer` parameter documentation states it "receives the current instance, field and value" but fails to mention that `inst` and `field` can be `None` for nested non-attrs values (list items, dict keys/values). This causes user code that reasonably assumes these parameters are never `None` to crash with `AttributeError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

@attr.s
class Container:
    data = attr.ib()

@given(st.lists(st.integers(), min_size=1))
def test_value_serializer_inst_field_never_none(items):
    def serializer(inst, field, value):
        assert inst is not None, "inst should never be None per docs"
        assert field is not None, "field should never be None per docs"
        return value

    obj = Container(data=items)
    attr.asdict(obj, recurse=True, value_serializer=serializer)
```

**Failing input**: `items = [1]`

## Reproducing the Bug

```python
import attr

@attr.s
class Container:
    data = attr.ib()

def serializer_using_field(inst, field, value):
    return f"{field.name}={value}"

obj = Container(data=[1, 2, 3])
result = attr.asdict(obj, recurse=True, value_serializer=serializer_using_field)
```

**Output**:
```
AttributeError: 'NoneType' object has no attribute 'name'
```

The crash occurs because `value_serializer` is called with `(None, None, value)` for each integer in the list, but the user's serializer reasonably expects `field` to never be `None` based on the documentation.

## Why This Is A Bug

The `asdict()` docstring (lines 44-48 in `_funcs.py`) states:

```
value_serializer (typing.Callable | None):
    A hook that is called for every attribute or dict key/value.  It
    receives the current instance, field and value and must return the
    (updated) value.
```

This documentation is **incomplete and misleading** because:

1. It says "receives the current instance, field and value" without mentioning these can be `None`
2. A reasonable user would implement a serializer that accesses `inst` or `field` properties
3. This causes crashes when processing nested non-attrs values

The actual implementation (lines 201-202 in `_funcs.py`):

```python
else:
    rv = val
    if value_serializer is not None:
        rv = value_serializer(None, None, rv)
```

This is called for:
- Items in nested lists/tuples/sets
- Keys and values in nested dicts
- Any other nested non-attrs primitives

## Fix

Update the docstring to accurately document the actual behavior:

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -43,9 +43,12 @@ def asdict(
             ordered dictionaries instead of normal Python dictionaries, pass in
             ``collections.OrderedDict``.

         value_serializer (typing.Callable | None):
             A hook that is called for every attribute or dict key/value.  It
-            receives the current instance, field and value and must return the
+            receives the current instance, field and value. For attrs class
+            attributes, ``inst`` is the attrs instance and ``field`` is the
+            ``Attribute`` object. For nested non-attrs values (e.g., items in
+            lists or dict keys/values), both ``inst`` and ``field`` are ``None``.
+            The hook must return the
             (updated) value.  The hook is run *after* the optional *filter* has
             been applied.
```

Alternatively, the implementation could be changed to never pass `None`, but this would be a breaking change for any code that currently handles the `None` case.