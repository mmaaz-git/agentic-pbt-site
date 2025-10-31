# Bug Report: django.utils.html.CountsDict Incorrect kwargs Unpacking

**Target**: `django.utils.html.CountsDict.__init__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`CountsDict.__init__` uses `*kwargs` instead of `**kwargs` when calling `super().__init__()`, preventing keyword arguments from being passed to the parent `dict` class and violating the expected dict API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils.html import CountsDict

@given(st.text(min_size=1), st.dictionaries(st.text(min_size=1), st.integers()))
def test_countsdict_accepts_kwargs_like_dict(word, kwargs_data):
    cd = CountsDict(word=word, **kwargs_data)
    for key, value in kwargs_data.items():
        assert cd[key] == value
```

**Failing input**: `CountsDict(word="hello", foo="bar")`

## Reproducing the Bug

```python
from django.utils.html import CountsDict

cd = CountsDict(word="hello", foo="bar")
```

**Output**:
```
TypeError: dict() argument after ** must be a mapping, not str
```

## Why This Is A Bug

`CountsDict` inherits from `dict` and declares `**kwargs` in its `__init__` signature, suggesting it should accept keyword arguments like any dict. However, line 281 uses `*kwargs` (single asterisk) instead of `**kwargs` (double asterisk) when calling the parent constructor:

```python
class CountsDict(dict):
    def __init__(self, *args, word, **kwargs):
        super().__init__(*args, *kwargs)  # Wrong: tries to unpack kwargs as positional args
```

This causes keyword arguments to be unpacked as positional arguments rather than keyword arguments, resulting in a TypeError. While the current codebase only uses `CountsDict(word=...)` (line 420), this violates the dict contract and prevents any future use with additional keyword arguments.

## Fix

```diff
--- a/django/utils/html.py
+++ b/django/utils/html.py
@@ -278,7 +278,7 @@ def smart_urlquote(url):

 class CountsDict(dict):
     def __init__(self, *args, word, **kwargs):
-        super().__init__(*args, *kwargs)
+        super().__init__(*args, **kwargs)
         self.word = word

     def __missing__(self, key):
```