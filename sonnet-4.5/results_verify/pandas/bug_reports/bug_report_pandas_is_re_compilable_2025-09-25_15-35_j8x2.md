# Bug Report: pandas.api.types.is_re_compilable Crashes on Invalid Patterns

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`is_re_compilable` crashes with `re.PatternError` when given invalid regex patterns instead of returning False as documented. The function signature indicates it returns bool, but it raises exceptions on inputs that cannot be compiled as regex patterns.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.api.types import is_re_compilable
import re

@given(st.text())
@settings(max_examples=500)
def test_is_re_compilable_consistency(pattern):
    try:
        re.compile(pattern)
        can_compile = True
    except re.error:
        can_compile = False

    result = is_re_compilable(pattern)

    if can_compile:
        assert result, f"Pattern '{pattern}' compiles successfully but is_re_compilable returned False"
```

**Failing input**: `pattern='\\C'` (and many others like `'['`, `')'`, `'?'`, `'*'`, `'\\1'`)

## Reproducing the Bug

```python
from pandas.api.types import is_re_compilable

result = is_re_compilable('\\C')
```

Output:
```
Traceback (most recent call last):
  ...
re.PatternError: bad escape \C at position 0
```

Other crashing inputs:
- `'['` → `re.PatternError: unterminated character set at position 0`
- `')'` → `re.PatternError: unbalanced parenthesis at position 0`
- `'?'` → `re.PatternError: nothing to repeat at position 0`
- `'*'` → `re.PatternError: nothing to repeat at position 0`
- `'\\1'` → `re.PatternError: invalid group reference 1 at position 1`

## Why This Is A Bug

The function signature is `is_re_compilable(obj) -> 'bool'`, and the docstring states it should "Check if the object can be compiled into a regex pattern instance." This clearly indicates the function should return a boolean value (True if compilable, False if not), not raise an exception.

The purpose of such a function is to safely check whether a string is a valid regex pattern without crashing. Users expect to use this as a guard:

```python
if is_re_compilable(user_input):
    pattern = re.compile(user_input)
else:
    # handle invalid pattern
```

But currently, it crashes on invalid patterns, defeating its purpose.

## Fix

```diff
def is_re_compilable(obj) -> bool:
    """
    Check if the object can be compiled into a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
    """
+   try:
        re.compile(obj)
+       return True
+   except (re.error, TypeError):
+       return False
-   return True
```

The fix wraps the `re.compile()` call in a try-except block to catch `re.error` (which includes `re.PatternError`) and return False instead of propagating the exception.