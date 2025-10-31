# Bug Report: llm.utils.truncate_string Violates Length Contract

**Target**: `llm.utils.truncate_string`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `truncate_string` function violates its documented contract when `max_length < 3`, returning strings that exceed the specified maximum length. The function's docstring states "max_length: Maximum length of the result string", but for values 1 and 2, it returns strings much longer than the specified maximum.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length
```

**Failing input**: `text="hello world", max_length=1`

## Reproducing the Bug

```python
from llm.utils import truncate_string

result = truncate_string("hello world", 1)
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max: 1")
```

Output:
```
Result: 'hello worl...'
Length: 13
Expected max: 1
```

Additional failing cases:
```python
truncate_string("test", 1)
truncate_string("test", 2)
```

## Why This Is A Bug

The function's docstring explicitly states that `max_length` is the "Maximum length of the result string". However, when `max_length < 3`, the function returns strings that violate this contract:

- `max_length=1` returns strings like "hello worl..." (length 13)
- `max_length=2` returns strings like "hello world..." (length 14)

The bug occurs in the final else branch:
```python
return text[: max_length - 3] + "..."
```

When `max_length < 3`, the slice `text[: max_length - 3]` becomes `text[:-2]` or `text[:-1]`, which can include most of the original string, plus the 3-character "..." suffix.

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -473,5 +473,8 @@ def truncate_string(
         cutoff = (max_length - 5) // 2
         return text[:cutoff] + "... " + text[-cutoff:]
     else:
-        # Fall back to simple truncation for very small max_length
-        return text[: max_length - 3] + "..."
+        if max_length < 3:
+            return text[:max_length]
+        else:
+            return text[: max_length - 3] + "..."
```