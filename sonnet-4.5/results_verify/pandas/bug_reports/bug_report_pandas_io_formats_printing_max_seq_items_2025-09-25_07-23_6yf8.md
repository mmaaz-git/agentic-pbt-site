# Bug Report: pandas.io.formats.printing ignores max_seq_items=0

**Target**: `pandas.io.formats.printing._pprint_seq` and `pprint_thing`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `max_seq_items=0` is passed to `_pprint_seq` or `pprint_thing`, the parameter is ignored and all items are printed instead of zero items. This happens because the code uses an `or` chain that treats 0 as falsy.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.io.formats.printing import _pprint_seq


@given(st.lists(st.integers(), min_size=1, max_size=20))
def test_pprint_seq_max_seq_items_zero(seq):
    """
    Property: When max_seq_items=0, the sequence should be truncated to 0 items.
    """
    result = _pprint_seq(seq, max_seq_items=0)
    assert '...' in result or result in ('[]', '()', '{}'), \
        f"max_seq_items=0 should show no items or ellipsis, got: {result}"
```

**Failing input**: `seq=[0]`
**Expected**: `'[]'` or `'[...]'`
**Actual**: `'[0]'`

## Reproducing the Bug

```python
from pandas.io.formats.printing import _pprint_seq, pprint_thing
import pandas as pd

seq = [1, 2, 3, 4, 5]

result = _pprint_seq(seq, max_seq_items=0)
print(f"_pprint_seq with max_seq_items=0: {result}")
print(f"Expected: [] or [...], Got: {result}")

result = pprint_thing(seq, max_seq_items=0)
print(f"pprint_thing with max_seq_items=0: {result}")
print(f"Expected: [] or [...], Got: {result}")

pd.set_option('display.max_seq_items', 0)
result = pprint_thing(seq)
print(f"pprint_thing with option set to 0: {result}")
print(f"Expected: [] or [...], Got: {result}")
```

Output:
```
_pprint_seq with max_seq_items=0: [1, 2, 3, 4, 5]
Expected: [] or [...], Got: [1, 2, 3, 4, 5]
pprint_thing with max_seq_items=0: [1, 2, 3, 4, 5]
Expected: [] or [...], Got: [1, 2, 3, 4, 5]
pprint_thing with option set to 0: [1, 2, 3, 4, 5]
Expected: [] or [...], Got: [1, 2, 3, 4, 5]
```

## Why This Is A Bug

The pandas option documentation states: "no more than `max_seq_items` will be printed". When `max_seq_items=0`, this should mean zero items are printed, but instead all items are printed.

The bug is on line 116 of printing.py:

```python
if max_seq_items is False:
    nitems = len(seq)
else:
    nitems = max_seq_items or get_option("max_seq_items") or len(seq)
```

The `or` chain treats 0 as falsy, so `max_seq_items=0` becomes `get_option("max_seq_items") or len(seq)`. If the option is also 0, it becomes `len(seq)`, meaning "print all items".

This violates the user's explicit request to limit output to 0 items.

## Fix

```diff
--- a/pandas/io/formats/printing.py
+++ b/pandas/io/formats/printing.py
@@ -113,7 +113,10 @@ def _pprint_seq(

     if max_seq_items is False:
         nitems = len(seq)
+    elif max_seq_items is not None:
+        nitems = max_seq_items
     else:
-        nitems = max_seq_items or get_option("max_seq_items") or len(seq)
+        option_val = get_option("max_seq_items")
+        nitems = option_val if option_val is not None else len(seq)

     s = iter(seq)
```