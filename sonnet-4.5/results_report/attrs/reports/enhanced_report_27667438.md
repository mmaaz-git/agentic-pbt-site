# Bug Report: attrs asdict Namedtuple Crash in Nested Collections

**Target**: `attr.asdict` (specifically `attr._funcs._asdict_anything`)
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `asdict()` processes namedtuples nested inside collections (lists, dicts, etc.) with `retain_collection_types=True`, it crashes with a TypeError due to incorrect constructor call syntax for namedtuples.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
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
@settings(max_examples=100)
def test_namedtuple_in_nested_collections(dict_items):
    if not dict_items or not dict_items[0]:
        return  # Skip empty cases

    nested_dict = {k: Point(v[0], v[1]) for k, v in list(dict_items[0].items())[:1]}
    obj = Container(data=nested_dict)
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    for key, point in nested_dict.items():
        assert isinstance(result['data'][key], Point)

if __name__ == "__main__":
    test_namedtuple_in_nested_collections()
```

<details>

<summary>
**Failing input**: `dict_items=[{'0': (0, 0)}]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 28, in <module>
    test_namedtuple_in_nested_collections()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 12, in test_namedtuple_in_nested_collections
    st.text(min_size=1, max_size=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 23, in test_namedtuple_in_nested_collections
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 106, in asdict
    rv[a.name] = df(
                 ~~^
        (
        ^
    ...<17 lines>...
        for kk, vv in v.items()
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 116, in <genexpr>
    _asdict_anything(
    ~~~~~~~~~~~~~~~~^
        vv,
        ^^^
    ...<4 lines>...
        value_serializer=value_serializer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ),
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 163, in _asdict_anything
    rv = cf(
        [
    ...<9 lines>...
        ]
    )
TypeError: Point.__new__() missing 1 required positional argument: 'y'
Falsifying example: test_namedtuple_in_nested_collections(
    dict_items=[{'0': (0, 0)}],
)
```
</details>

## Reproducing the Bug

```python
import attr
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

@attr.s
class Container:
    data = attr.ib()

# Create an instance with a namedtuple nested in a dictionary
obj = Container(data={'key': Point(1, 2)})

# This should work but crashes
try:
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
    print(f"Success: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: Point.__new__() missing 1 required positional argument: 'y'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/repo.py", line 15, in <module>
    result = attr.asdict(obj, recurse=True, retain_collection_types=True)
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 106, in asdict
    rv[a.name] = df(
                 ~~^
        (
        ^
    ...<17 lines>...
        for kk, vv in v.items()
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 116, in <genexpr>
    _asdict_anything(
    ~~~~~~~~~~~~~~~~^
        vv,
        ^^^
    ...<4 lines>...
        value_serializer=value_serializer,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ),
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/attr/_funcs.py", line 163, in _asdict_anything
    rv = cf(
        [
    ...<9 lines>...
        ]
    )
TypeError: Point.__new__() missing 1 required positional argument: 'y'
TypeError: Point.__new__() missing 1 required positional argument: 'y'
```
</details>

## Why This Is A Bug

The `retain_collection_types=True` parameter is specifically documented to preserve the original collection types when recursing through data structures. When this flag is set, the function should maintain namedtuples as namedtuples rather than converting them to regular tuples or lists.

The bug occurs because namedtuples have a unique constructor signature that differs from regular tuples. While regular tuples can be constructed with `tuple([1, 2])`, namedtuples require either positional arguments `Point(1, 2)` or unpacking `Point(*[1, 2])`.

The main `asdict` function already handles this correctly for direct attribute values (lines 96-103 in `attr/_funcs.py`), using a try/except block to catch the TypeError and retry with unpacked arguments. However, the helper function `_asdict_anything`, which processes nested values, lacks this same workaround. This inconsistency means that namedtuples work when they're direct attribute values but fail when nested inside dictionaries or lists.

## Relevant Context

The bug is in attrs version 25.3.0 and affects the `_asdict_anything` helper function at line 163 of `/attr/_funcs.py`. The main `asdict` function already has the correct namedtuple handling code at lines 96-103:

```python
try:
    rv[a.name] = cf(items)
except TypeError:
    if not issubclass(cf, tuple):
        raise
    # Workaround for TypeError: cf.__new__() missing 1 required
    # positional argument (which appears, for a namedturle)
    rv[a.name] = cf(*items)
```

The same pattern appears in the `astuple` function at lines 283-290, showing this is a known issue that has been addressed elsewhere in the codebase but was overlooked in `_asdict_anything`.

GitHub repository: https://github.com/python-attrs/attrs
Affected file: https://github.com/python-attrs/attrs/blob/main/src/attr/_funcs.py

## Proposed Fix

Apply the same namedtuple workaround that exists in `asdict` (lines 96-103) to `_asdict_anything`:

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -161,7 +161,7 @@ def _asdict_anything(
         else:
             cf = list

-        rv = cf(
+        items = [
             [
                 _asdict_anything(
                     i,
@@ -173,7 +173,15 @@ def _asdict_anything(
                 )
                 for i in val
             ]
-        )
+        ]
+        try:
+            rv = cf(items)
+        except TypeError:
+            if not issubclass(cf, tuple):
+                raise
+            # Workaround for TypeError: cf.__new__() missing 1 required
+            # positional argument (which appears, for a namedtuple)
+            rv = cf(*items)
     elif isinstance(val, dict):
         df = dict_factory
         rv = df(
```