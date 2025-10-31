# Bug Report: pandas.core.strings rsplit() Missing regex Parameter

**Target**: `pandas.core.strings.accessor.StringMethods.rsplit`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `rsplit()` method lacks the `regex` parameter that its mirror operation `split()` has, creating an API inconsistency between two closely related methods that perform the same operation from different directions.

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

**Failing input**: All inputs fail - `rsplit` signature is `(self, pat=None, *, n=-1, expand: bool = False)` while `split` signature is `(self, pat: str | re.Pattern | None = None, *, n=-1, expand: bool = False, regex: bool | None = None)`

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

The `split()` method accepts a `regex` parameter (added in pandas 1.4.0) to control whether the pattern is treated as a regular expression or a literal string. This parameter provides important control:
- `regex=True`: Treat pattern as regex
- `regex=False`: Treat pattern as literal string
- `regex=None` (default): Use heuristic (single char → literal, multi-char → regex)

The `rsplit()` method, documented as the right-to-left counterpart of `split()`, completely lacks this parameter. Examining the implementation reveals:
- `split()` has sophisticated regex handling in `_str_split()`
- `rsplit()` simply calls Python's `str.rsplit()` in `_str_rsplit()`, which only supports literal strings

This creates an API inconsistency where:
1. Users can control regex behavior for `split()` but not for `rsplit()`
2. `rsplit()` always treats patterns as literal strings, while `split()` has regex support
3. Code using `split(regex=False)` cannot be adapted to `rsplit()` by simply changing the method name
4. Users cannot use regex patterns with `rsplit()` at all

## Fix

Add regex parameter and support to `rsplit()`. Modify the signature in `accessor.py`:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -1000,7 +1000,7 @@
     )
     @forbid_nonstring_types(["bytes"])
-    def rsplit(self, pat=None, *, n=-1, expand: bool = False):
+    def rsplit(self, pat=None, *, n=-1, expand: bool = False, regex: bool | None = None):
         result = self._data.array._str_rsplit(pat, n=n)
```

And update `_str_rsplit()` in `object_array.py` to mirror the logic from `_str_split()` for handling the `regex` parameter.