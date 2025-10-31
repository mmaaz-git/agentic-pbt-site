# Bug Report: attrs asdict Namedtuple in Nested Collections

**Target**: `attr.asdict` (specifically `attr._funcs._asdict_anything`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `asdict()` processes namedtuples nested inside collections (lists, dicts, etc.) with `retain_collection_types=True`, it crashes with a TypeError due to incorrect constructor call syntax.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from collections import namedtuple
import attr

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

@given(st.lists(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.tuples(st.integers(), st.integers()),
    min_size=1, max_size=5
)))
def test_namedtuple_in_nested_collections(dict_items):
    nested_dict = {k: Point(v[0], v[1]) for k, v in list(dict_items[0].items())[:1]}
    obj = Container(data=nested_dict)
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    for key, point in nested_dict.items():
        assert isinstance(result['data'][key], Point)
```

**Failing input**: `Container(data={'a': Point(1, 2)})`

## Reproducing the Bug

```python
import attr
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

obj = Container(data={'key': Point(1, 2)})
result = attr.asdict(obj, recurse=True, retain_collection_types=True)
```

**Error output:**
```
TypeError: Point.__new__() missing 1 required positional argument: 'y'
```

## Why This Is A Bug

The function `attr.asdict` has a workaround for namedtuples when they are direct field values (lines 96-103 in `attr/_funcs.py`):

```python
try:
    rv[a.name] = cf(items)
except TypeError:
    if not issubclass(cf, tuple):
        raise
    # Workaround for namedtuple
    rv[a.name] = cf(*items)
```

However, the helper function `_asdict_anything` (which handles nested values) lacks this workaround. At line 163, it calls:

```python
rv = cf([
    _asdict_anything(i, ...) for i in val
])
```

For namedtuples, `cf` is the namedtuple class (e.g., `Point`), and calling `Point([1, 2])` fails because namedtuples require `Point(1, 2)` or `Point(*[1, 2])`.

This bug affects any namedtuple that appears:
- As values in nested dictionaries
- As elements in nested lists
- In any nested collection when `retain_collection_types=True`

## Fix

Apply the same namedtuple workaround that exists in `asdict` (lines 96-103) to `_asdict_anything`:

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -163,7 +163,7 @@ def _asdict_anything(
         else:
             cf = list

-        rv = cf(
+        items = (
             [
                 _asdict_anything(
                     i,
@@ -175,6 +175,14 @@ def _asdict_anything(
                 for i in val
             ]
         )
+        try:
+            rv = cf(items)
+        except TypeError:
+            if not issubclass(cf, tuple):
+                raise
+            # Workaround for namedtuple (same as lines 96-103)
+            rv = cf(*items)
+
     elif isinstance(val, dict):
         df = dict_factory
         rv = df(
```