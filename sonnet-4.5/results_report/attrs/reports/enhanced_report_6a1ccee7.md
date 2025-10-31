# Bug Report: attrs asdict() Crashes on Namedtuples in Nested Collections

**Target**: `attr._funcs._asdict_anything`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `asdict()` function crashes with a TypeError when processing namedtuples that are nested inside collections when using `retain_collection_types=True`. While the main `asdict()` function has a workaround for namedtuples, the helper function `_asdict_anything()` lacks this same workaround.

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

<details>

<summary>
**Failing input**: `coords=[(0, 0)]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 23, in <module>
    test_asdict_with_list_of_namedtuples()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 13, in test_asdict_with_list_of_namedtuples
    def test_asdict_with_list_of_namedtuples(coords):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 16, in test_asdict_with_list_of_namedtuples
    result = asdict(c, retain_collection_types=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 86, in asdict
    _asdict_anything(
    ~~~~~~~~~~~~~~~~^
        i,
        ^^
    ...<4 lines>...
        value_serializer=value_serializer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 163, in _asdict_anything
    rv = cf(
        [
    ...<9 lines>...
        ]
    )
TypeError: Point.__new__() missing 1 required positional argument: 'y'
Falsifying example: test_asdict_with_list_of_namedtuples(
    coords=[(0, 0)],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import attr
from attr import asdict
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    points = attr.ib()

# Create a container with a list containing a namedtuple
c = Container(points=[Point(1, 2)])

# This should crash with TypeError
result = asdict(c, retain_collection_types=True)
print(f"Result: {result}")
```

<details>

<summary>
TypeError when calling asdict() with namedtuple in list
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/repo.py", line 15, in <module>
    result = asdict(c, retain_collection_types=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 86, in asdict
    _asdict_anything(
    ~~~~~~~~~~~~~~~~^
        i,
        ^^
    ...<4 lines>...
        value_serializer=value_serializer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 163, in _asdict_anything
    rv = cf(
        [
    ...<9 lines>...
        ]
    )
TypeError: Point.__new__() missing 1 required positional argument: 'y'
```
</details>

## Why This Is A Bug

This bug violates expected behavior because there is an inconsistency in how namedtuples are handled within the attrs library. The main `asdict()` function (lines 96-103 in `_funcs.py`) contains a specific workaround for namedtuples with a try-except block that catches TypeError and uses `cf(*items)` instead of `cf(items)` when the collection factory is a tuple subclass. The comment at line 101-102 explicitly states: "Workaround for TypeError: cf.__new__() missing 1 required positional argument (which appears, for a namedturle)".

The same workaround exists in the `astuple()` function (lines 283-290) with an identical comment at lines 288-289. However, the `_asdict_anything()` helper function (lines 134-204), which is called recursively to process nested structures, lacks this workaround. At line 163-175, it directly calls `cf([...])` without the try-except protection, causing crashes when encountering namedtuples.

The parameter `retain_collection_types=True` is documented as "Do not convert to `list` when encountering an attribute whose type is `tuple` or `set`" (lines 40-42), implying that the original collection types should be preserved. Since namedtuples are tuple subclasses, they should be preserved and handled correctly when this parameter is True.

## Relevant Context

The bug only manifests when all of these conditions are met:
1. Using `asdict()` function (not `astuple()`)
2. With `retain_collection_types=True` parameter
3. On an attrs class containing namedtuples
4. Where the namedtuples are inside a collection (list, set, tuple, etc.)

Without `retain_collection_types=True`, the function works correctly because namedtuples are converted to regular lists. The inconsistency between the main function and its helper is clearly unintentional, as evidenced by the existing workarounds in two other locations in the same file.

There's also a minor typo in the comments: "namedturle" should be "namedtuple" (lines 102 and 289).

## Proposed Fix

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -160,7 +160,7 @@ def _asdict_anything(
         else:
             cf = list

-        rv = cf(
-            [
+        items = [
                 _asdict_anything(
                     i,
                     is_key=False,
@@ -171,8 +171,15 @@ def _asdict_anything(
                     value_serializer=value_serializer,
                 )
                 for i in val
-            ]
-        )
+        ]
+        try:
+            rv = cf(items)
+        except TypeError:
+            if not issubclass(cf, tuple):
+                raise
+            # Workaround for TypeError: cf.__new__() missing 1 required
+            # positional argument (which appears for a namedtuple)
+            rv = cf(*items)
     elif isinstance(val, dict):
         df = dict_factory
         rv = df(
```