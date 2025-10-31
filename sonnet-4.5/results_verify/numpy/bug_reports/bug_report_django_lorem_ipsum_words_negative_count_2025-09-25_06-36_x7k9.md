# Bug Report: django.utils.lorem_ipsum.words Negative Count Handling

**Target**: `django.utils.lorem_ipsum.words()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `words()` function incorrectly handles negative count values when `common=True`, returning unexpected words instead of empty string or raising an error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils import lorem_ipsum

@given(st.integers(min_value=-100, max_value=-1))
def test_words_negative_count_invariant(count):
    result = lorem_ipsum.words(count, common=True)
    word_count = len(result.split()) if result else 0
    assert word_count == 0, (
        f"words({count}, common=True) should return empty string or raise ValueError, "
        f"but returned {word_count} words"
    )
```

**Failing input**: `count=-1, common=True`

## Reproducing the Bug

```python
from django.utils.lorem_ipsum import words

result = words(-1, common=True)
actual_count = len(result.split())

print(f"words(-1, common=True) returned {actual_count} words")
print(f"Expected: 0 words or ValueError")
print(f"Actual: {actual_count} words")
```

**Output:**
```
words(-1, common=True) returned 18 words
Expected: 0 words or ValueError
Actual: 18 words
```

## Why This Is A Bug

The function's docstring states it should "Return a string of `count` lorem ipsum words". Negative counts are invalid input - the function should either:
1. Return an empty string (0 words)
2. Raise `ValueError` for invalid input

Instead, when `count=-1` and `common=True`, the function returns 18 words due to Python's negative list slicing behavior (`word_list[:-1]` returns all but the last element).

This violates the function's contract: `words(-1)` should not return 18 words.

## Fix

```diff
--- a/django/utils/lorem_ipsum.py
+++ b/django/utils/lorem_ipsum.py
@@ -269,6 +269,8 @@ def paragraphs(count, common=True):
 def words(count, common=True):
     """
     Return a string of `count` lorem ipsum words separated by a single space.

     If `common` is True, then the first 19 words will be the standard
     'lorem ipsum' words. Otherwise, all words will be selected randomly.
     """
+    if count < 0:
+        return ""
     word_list = list(COMMON_WORDS) if common else []
     c = len(word_list)
     if count > c:
```

Alternative fix (raise exception for invalid input):

```diff
--- a/django/utils/lorem_ipsum.py
+++ b/django/utils/lorem_ipsum.py
@@ -269,6 +269,8 @@ def paragraphs(count, common=True):
 def words(count, common=True):
     """
     Return a string of `count` lorem ipsum words separated by a single space.

     If `common` is True, then the first 19 words will be the standard
     'lorem ipsum' words. Otherwise, all words will be selected randomly.
     """
+    if count < 0:
+        raise ValueError("count must be non-negative")
     word_list = list(COMMON_WORDS) if common else []
     c = len(word_list)
     if count > c:
```