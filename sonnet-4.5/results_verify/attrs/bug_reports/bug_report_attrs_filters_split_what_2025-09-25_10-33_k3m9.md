# Bug Report: attrs.filters._split_what Generator Exhaustion

**Target**: `attr.filters._split_what`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_split_what` function silently loses data when passed a generator instead of a list, because it iterates over the input three times without converting it to a reusable collection first.

## Property-Based Test

```python
from attr.filters import _split_what
from hypothesis import given, strategies as st

@given(st.lists(st.one_of(
    st.sampled_from([int, str, float]),
    st.text(min_size=1, max_size=20)
), min_size=1, max_size=20))
def test_split_what_generator_vs_list(items):
    gen_classes, gen_names, gen_attrs = _split_what(x for x in items)
    list_classes, list_names, list_attrs = _split_what(items)

    assert gen_classes == list_classes
    assert gen_names == list_names
    assert gen_attrs == list_attrs
```

**Failing input**: `items=['0']` (any list with at least one string)

## Reproducing the Bug

```python
from attr.filters import _split_what

items = [int, str, "name1", "name2", float]
gen = (x for x in items)

classes, names, attrs = _split_what(gen)

print(f"Classes: {classes}")
print(f"Names: {names}")

assert classes == frozenset({int, str, float})
assert names == frozenset({"name1", "name2"})
```

**Output:**
```
Classes: frozenset({<class 'int'>, <class 'str'>, <class 'float'>})
Names: frozenset()
AssertionError: Names should be {'name1', 'name2'} but is empty
```

## Why This Is A Bug

The function iterates over the `what` parameter three times (lines 15-17 in filters.py):

```python
def _split_what(what):
    return (
        frozenset(cls for cls in what if isinstance(cls, type)),      # 1st iteration
        frozenset(cls for cls in what if isinstance(cls, str)),       # 2nd iteration - EMPTY!
        frozenset(cls for cls in what if isinstance(cls, Attribute)), # 3rd iteration - EMPTY!
    )
```

When `what` is a generator, the first comprehension exhausts it, leaving nothing for the second and third iterations. This violates the expected behavior:
1. The function doesn't document this limitation
2. Generators are common Python iterables (e.g., from `filter()`, generator expressions)
3. The failure is silent - no error, just incorrect empty results

This bug affects real usage because:
- `include()` and `exclude()` filters call `_split_what` internally
- Users might naturally pass generator expressions like `include(*(f for f in fields if condition))`

## Fix

```diff
diff --git a/attr/filters.py b/attr/filters.py
index 1234567..abcdefg 100644
--- a/attr/filters.py
+++ b/attr/filters.py
@@ -10,6 +10,7 @@ from ._make import Attribute
 def _split_what(what):
     """
     Returns a tuple of `frozenset`s of classes and attributes.
     """
+    what = tuple(what)
     return (
         frozenset(cls for cls in what if isinstance(cls, type)),
         frozenset(cls for cls in what if isinstance(cls, str)),
```

This fix converts the input to a tuple once, ensuring it can be iterated multiple times regardless of whether it's a generator, list, or other iterable.