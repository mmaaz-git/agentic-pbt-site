# Bug Report: pandas.io.formats.printing.adjoin - Inconsistent Application of strlen Parameter to Input Lists

**Target**: `pandas.io.formats.printing.adjoin`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `adjoin` function incorrectly applies the custom `strlen` parameter only to the first n-1 input lists, using the built-in `len()` function for the last list instead, breaking unicode text formatting for the last column.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.printing import adjoin


@given(st.lists(st.text(min_size=1), min_size=1), st.lists(st.text(min_size=1), min_size=1))
def test_adjoin_uses_strlen_consistently(list1, list2):
    call_count = {"count": 0}

    def counting_strlen(s):
        call_count["count"] += 1
        return len(s)

    adjoin(1, list1, list2, strlen=counting_strlen)

    total_strings = len(list1) + len(list2)
    assert call_count["count"] >= total_strings, f"Expected at least {total_strings} calls to strlen, but got {call_count['count']}"


# Run the test
test_adjoin_uses_strlen_consistently()
```

<details>

<summary>
**Failing input**: `test_adjoin_uses_strlen_consistently(list1=['0'], list2=['0'])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 20, in <module>
    test_adjoin_uses_strlen_consistently()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 6, in test_adjoin_uses_strlen_consistently
    def test_adjoin_uses_strlen_consistently(list1, list2):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 16, in test_adjoin_uses_strlen_consistently
    assert call_count["count"] >= total_strings, f"Expected at least {total_strings} calls to strlen, but got {call_count['count']}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected at least 2 calls to strlen, but got 1
Falsifying example: test_adjoin_uses_strlen_consistently(
    # The test always failed when commented parts were varied together.
    list1=['0'],  # or any other generated value
    list2=['0'],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pandas.io.formats.printing import adjoin


def custom_strlen(s):
    print(f"custom_strlen called with: '{s}'")
    return len(s) + 10


# Test case demonstrating the inconsistency
result = adjoin(1, ["a", "bb"], ["c", "dd"], strlen=custom_strlen)
print("\nResult:")
print(repr(result))
print("\nFormatted output:")
print(result)

print("\n" + "="*50)
print("Notice: custom_strlen was only called for the first list items ('a' and 'bb'),")
print("but NOT for the second list items ('c' and 'dd').")
print("This demonstrates the bug where strlen is not applied consistently to all lists.")
```

<details>

<summary>
Output demonstrating the bug - custom_strlen is not called for the last list
</summary>
```
custom_strlen called with: 'a'
custom_strlen called with: 'bb'

Result:
'a            c \nbb           dd'

Formatted output:
a            c
bb           dd

==================================================
Notice: custom_strlen was only called for the first list items ('a' and 'bb'),
but NOT for the second list items ('c' and 'dd').
This demonstrates the bug where strlen is not applied consistently to all lists.
```
</details>

## Why This Is A Bug

The function's docstring explicitly documents that the `strlen` parameter is a "function used to calculate the length of each str. Needed for unicode handling." However, the implementation contradicts this documentation by only applying `strlen` to all lists except the last one.

In the source code at `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/formats/printing.py`:

- **Line 51**: `lengths = [max(map(strlen, x)) + space for x in lists[:-1]]` - correctly uses the custom `strlen` function for all lists except the last
- **Line 53**: `lengths.append(max(map(len, lists[-1]))` - incorrectly uses the built-in `len()` function for the last list

This inconsistency defeats the documented purpose of the `strlen` parameter. When handling unicode text with display widths different from character counts (such as East Asian characters that take up 2 character widths), the last column will be misaligned because it uses character count instead of display width.

## Relevant Context

The pandas library includes a `_EastAsianTextAdjustment` class (lines 528-564 in the same file) that provides custom length calculation for East Asian characters. This class's `adjoin` method (line 525) specifically passes its custom `len` method as the `strlen` parameter to handle unicode width calculations correctly. The bug would cause this functionality to fail for the last column of any table.

The function is used internally throughout pandas for formatting output in multiple modules:
- `pandas/io/formats/string.py`
- `pandas/io/formats/format.py`
- `pandas/io/pytables.py`
- `pandas/core/indexes/multi.py`

This means the bug affects not just direct users of the function but also various pandas formatting operations that rely on it.

## Proposed Fix

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