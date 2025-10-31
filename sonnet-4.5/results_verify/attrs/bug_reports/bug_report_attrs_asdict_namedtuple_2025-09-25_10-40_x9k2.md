# Bug Report: attrs asdict() Crashes on Namedtuples with retain_collection_types

**Target**: `attr._funcs._asdict_anything`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `asdict()` function crashes with a TypeError when processing namedtuples inside collections with `retain_collection_types=True`. The main `asdict` function has a workaround for namedtuples, but the helper `_asdict_anything` function lacks the same workaround, causing crashes on nested structures.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from collections import namedtuple
import attr
from attr import asdict

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

@given(st.lists(st.tuples(st.integers(), st.integers()), min_size=1, max_size=5))
def test_asdict_with_list_of_namedtuples(coords):
    points = [Point(x, y) for x, y in coords]
    c = Container(points=points)
    result = asdict(c, retain_collection_types=True)

    assert isinstance(result['points'], list)
    assert len(result['points']) == len(points)
```

**Failing input**: `coords=[(0, 0)]`

## Reproducing the Bug

```python
import attr
from attr import asdict
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

c = Container(points=[Point(1, 2)])
result = asdict(c, retain_collection_types=True)
```

**Output:**
```
TypeError: Point.__new__() missing 1 required positional argument: 'y'
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 163, in _asdict_anything
    rv = cf([...])
```

## Why This Is A Bug

The main `asdict` function has special handling for namedtuples (lines 96-103 in `_funcs.py`):

```python
try:
    rv[a.name] = cf(items)
except TypeError:
    if not issubclass(cf, tuple):
        raise
    rv[a.name] = cf(*items)  # Workaround for namedtuples
```

However, the `_asdict_anything` helper function (lines 163-175) lacks this workaround:

```python
rv = cf([...])  # This fails for namedtuples!
```

This inconsistency causes crashes when namedtuples appear in nested collections. The function should handle namedtuples consistently in both locations.

## Fix

Apply the same namedtuple workaround in `_asdict_anything` as exists in the main `asdict` function:

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -160,7 +160,14 @@ def _asdict_anything(
         else:
             cf = list

-        rv = cf(
+        items = (
             [
                 _asdict_anything(
                     i,
@@ -172,7 +179,14 @@ def _asdict_anything(
                 )
                 for i in val
             ]
         )
+        try:
+            rv = cf(items)
+        except TypeError:
+            if not issubclass(cf, tuple):
+                raise
+            # Workaround for namedtuples
+            rv = cf(*items)
     elif isinstance(val, dict):
         df = dict_factory
         rv = df(
```