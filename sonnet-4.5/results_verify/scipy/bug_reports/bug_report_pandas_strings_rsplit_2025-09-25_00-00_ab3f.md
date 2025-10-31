# Bug Report: pandas.core.strings rsplit() Missing regex Parameter

**Target**: `pandas.core.strings.accessor.StringMethods.rsplit`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `rsplit()` method lacks the `regex` parameter that its mirror operation `split()` has, creating an API inconsistency between two closely related methods.

## Property-Based Test

```python
import pandas as pd
import inspect

def test_split_rsplit_have_same_parameters():
    split_sig = inspect.signature(pd.core.strings.accessor.StringMethods.split)
    rsplit_sig = inspect.signature(pd.core.strings.accessor.StringMethods.rsplit)

    split_params = set(split_sig.parameters.keys())
    rsplit_params = set(rsplit_sig.parameters.keys())

    assert 'regex' in split_params
    assert 'regex' in rsplit_params
```

**Failing input**: Any test run will fail

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['a.b.c.d'])

print(s.str.split('.', regex=True).iloc[0])

print(s.str.split('.', regex=False).iloc[0])

print(s.str.rsplit('.').iloc[0])

try:
    s.str.rsplit('.', regex=False)
except TypeError as e:
    print(f"ERROR: {e}")
```

**Output:**
```
['', '', '', '', '', '', '', '']
['a', 'b', 'c', 'd']
['a', 'b', 'c', 'd']
ERROR: StringMethods.rsplit() got an unexpected keyword argument 'regex'
```

## Why This Is A Bug

The `split()` method accepts a `regex` parameter (added in pandas 1.4.0) to control whether the pattern is treated as a regular expression or a literal string. The `rsplit()` method, which is documented as a mirror operation that "splits from the right", lacks this parameter entirely.

This creates an API inconsistency where:
1. Users can control regex behavior for `split()` but not for `rsplit()`
2. `rsplit()` always treats patterns as literal strings (via Python's `str.rsplit()`), while `split()` has sophisticated regex support
3. Code that works with `split(regex=False)` cannot be easily adapted to use `rsplit()` by simply changing the method name

## Fix

Add the `regex` parameter to `rsplit()` method in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/strings/accessor.py` and implement regex support in `_str_rsplit()` in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/strings/object_array.py`.

The implementation in `_str_rsplit` should mirror the logic in `_str_split` to handle the `regex` parameter consistently.