# Bug Report: pandas.api.types.is_re_compilable Raises Instead of Returning False

**Target**: `pandas.api.types.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`is_re_compilable()` is supposed to return True if a string can be compiled as a regex and False otherwise. However, it raises `re.PatternError` for invalid regex patterns instead of returning False, violating its documented contract.

## Property-Based Test

```python
import pandas.api.types as pat
import re
from hypothesis import given, strategies as st


@given(st.text())
def test_is_re_compilable_should_not_raise(s):
    try:
        re.compile(s)
        can_compile = True
    except Exception:
        can_compile = False

    result = pat.is_re_compilable(s)

    assert result == can_compile, (
        f"is_re_compilable should match re.compile behavior without raising, "
        f"but got different behavior for {s!r}"
    )
```

**Failing input**: `s='('`

## Reproducing the Bug

```python
import pandas.api.types as pat

print(pat.is_re_compilable('.*'))

print(pat.is_re_compilable('('))
```

Expected output:
```
True
False
```

Actual output:
```
True
re.PatternError: missing ), unterminated subpattern at position 0
```

## Why This Is A Bug

1. **Violates API contract**: The function's docstring says it returns a bool indicating whether the object can be compiled as a regex. It should never raise an exception for string inputs.

2. **Inconsistent error handling**: The function catches `TypeError` but not `re.error` (or its subclass `re.PatternError`), leading to unexpected crashes for invalid regex patterns.

3. **Makes the function unreliable**: Users cannot safely use this function to check if a string is a valid regex without wrapping it in a try-except block, defeating its purpose.

## Fix

Catch all regex compilation errors, not just `TypeError`:

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
        Whether `obj` can be compiled as a regex pattern.

    Examples
    --------
    >>> from pandas.api.types import is_re_compilable
    >>> is_re_compilable(".*")
    True
    >>> is_re_compilable(1)
    False
    """
    try:
        re.compile(obj)
-   except TypeError:
+   except (TypeError, re.error):
        return False
    else:
        return True
```

Note: `re.error` is the base exception class for all regex compilation errors, including `PatternError`, `ValueError`, etc.