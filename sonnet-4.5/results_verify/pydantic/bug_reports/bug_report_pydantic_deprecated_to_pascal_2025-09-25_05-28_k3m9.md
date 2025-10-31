# Bug Report: pydantic.deprecated.decorator.to_pascal - Non-idempotent behavior

**Target**: `pydantic.deprecated.decorator.to_pascal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_pascal` function is not idempotent - applying it twice to the same input produces different results. Specifically, when the function converts a snake_case string to PascalCase, applying it again to the result incorrectly lowercases some characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.deprecated.decorator import to_pascal


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1))
def test_to_pascal_idempotent(snake):
    first = to_pascal(snake)
    second = to_pascal(first)
    assert first == second, f"to_pascal should be idempotent: to_pascal({snake!r}) = {first!r}, to_pascal({first!r}) = {second!r}"
```

**Failing input**: `'a_a'`

## Reproducing the Bug

```python
from pydantic.deprecated.decorator import to_pascal

assert to_pascal('a_a') == 'AA'
assert to_pascal('AA') == 'Aa'
assert to_pascal('a_a') != to_pascal(to_pascal('a_a'))
```

## Why This Is A Bug

A PascalCase converter should be idempotent - applying it to an already-PascalCase string should not modify the string. This is a fundamental expectation for case conversion utilities.

The bug occurs because the function uses `str.title()` which lowercases all characters except the first character of each word. When applied to an already-converted PascalCase string like 'AA', it incorrectly lowercases the second character to produce 'Aa'.

This affects real users who might:
1. Apply the function to strings that are already in PascalCase
2. Apply the function multiple times in a pipeline
3. Use the function on mixed-case inputs

## Fix

Replace `str.title()` with a solution that capitalizes the first letter of each underscore-separated word without lowercasing existing uppercase letters:

```diff
 def to_pascal(snake: str) -> str:
     """Convert a snake_case string to PascalCase.

     Args:
         snake: The string to convert.

     Returns:
         The PascalCase string.
     """
-    camel = snake.title()
-    return re.sub('([0-9A-Za-z])_(?=[0-9A-Z])', lambda m: m.group(1), camel)
+    def capitalize_word(word: str) -> str:
+        if not word:
+            return word
+        return word[0].upper() + word[1:]
+
+    parts = snake.split('_')
+    return ''.join(capitalize_word(part) for part in parts)
```