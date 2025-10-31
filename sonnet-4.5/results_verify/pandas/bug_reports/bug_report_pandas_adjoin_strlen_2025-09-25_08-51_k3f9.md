# Bug Report: pandas.io.formats.printing.adjoin strlen Inconsistency

**Target**: `pandas.io.formats.printing.adjoin`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `adjoin` function inconsistently applies the `strlen` parameter: it uses the custom `strlen` function for all lists except the last one, where it uses the builtin `len` function instead. This breaks the intended abstraction for handling unicode or custom string length calculations.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.formats.printing import adjoin


@given(
    lists=st.lists(
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
        min_size=2,
        max_size=5
    ),
    space=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=500)
def test_adjoin_strlen_consistency(lists, space):
    strlen_calls = []

    def tracking_strlen(s):
        strlen_calls.append(s)
        return len(s) + 1

    result = adjoin(space, *lists, strlen=tracking_strlen)

    all_strings = [s for lst in lists for s in lst]

    for s in all_strings:
        assert s in strlen_calls, f"strlen not called for '{s}'"
```

**Failing input**: Any input with 2 or more lists - the strings in the last list are never passed to the custom `strlen` function.

## Reproducing the Bug

```python
from pandas.io.formats.printing import adjoin


def my_strlen(s):
    print(f"my_strlen called with: '{s}'")
    return len(s) + 5


list1 = ["a", "b"]
list2 = ["x", "y"]
list3 = ["1", "2"]

result = adjoin(1, list1, list2, list3, strlen=my_strlen)
```

**Output**:
```
my_strlen called with: 'a'
my_strlen called with: 'b'
my_strlen called with: 'x'
my_strlen called with: 'y'
```

**Notice**: `my_strlen` is never called for list3 (the last list), even though it was provided as a parameter.

## Why This Is A Bug

The docstring for `adjoin` states that `strlen` is a "function used to calculate the length of each str. Needed for unicode handling." If a custom `strlen` is provided (e.g., to handle unicode width correctly), it should be applied consistently to all lists, not just lists[:-1].

The inconsistency occurs at line 51-53 of `pandas/io/formats/printing.py`:

```python
lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
# not the last one
lengths.append(max(map(len, lists[-1])))  # <-- Uses 'len' instead of 'strlen'
```

This means:
- Lists before the last: measured using `strlen`
- Last list: measured using builtin `len`

For unicode text with non-standard display widths, this could cause the last column to be incorrectly sized when using the `_EastAsianTextAdjustment` class which provides a custom `strlen`.

## Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -50,7 +50,7 @@ def adjoin(space: int, *lists: list[str], **kwargs) -> str:
     newLists = []
     lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
     # not the last one
-    lengths.append(max(map(len, lists[-1])))
+    lengths.append(max(map(strlen, lists[-1])))
     maxLen = max(map(len, lists))
     for i, lst in enumerate(lists):
         nl = justfunc(lst, lengths[i], mode="left")
```