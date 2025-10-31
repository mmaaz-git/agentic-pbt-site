# Bug Report: Cython.Compiler.TreePath Multiple Error Handling Bugs

**Target**: `Cython.Compiler.TreePath._build_path_iterator` and `parse_path_value`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The TreePath module has multiple error handling inconsistencies. Functions raise `StopIteration`, `KeyError`, and `AssertionError` for invalid inputs instead of uniformly raising `ValueError`. Additionally, `parse_path_value` uses `assert` statements for validation, which are disabled with `python -O`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Compiler.TreePath import _build_path_iterator, parse_path_value
import pytest


@settings(max_examples=1000)
@given(st.text(max_size=100))
def test_build_path_consistent_errors(path):
    try:
        result = _build_path_iterator(path)
        assert isinstance(result, list)
    except ValueError:
        pass
    except (StopIteration, KeyError, AssertionError) as e:
        pytest.fail(f"Bug: {type(e).__name__} instead of ValueError for '{path}'")


class MockNext:
    def __init__(self, token):
        self.token = token
        self.called = False

    def __call__(self):
        if self.called:
            raise StopIteration
        self.called = True
        return self.token


@given(
    st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=1),
    st.sampled_from(["'", '"']),
    st.sampled_from(["'", '"'])
)
def test_parse_value_mismatched_quotes(content, open_q, close_q):
    assume(open_q not in content and close_q not in content and open_q != close_q)
    token = (f"{open_q}{content}{close_q}", '')
    with pytest.raises((ValueError, AssertionError)):
        parse_path_value(MockNext(token))
```

**Failing inputs**:
- `_build_path_iterator('')` → `StopIteration` (should be `ValueError`)
- `_build_path_iterator('=')` → `KeyError: '='` (should be `ValueError`)
- `_build_path_iterator('/a')` → `KeyError: '/'` (should be `ValueError`)
- `parse_path_value` with `"'abc\""` → `AssertionError` (should be `ValueError`)

## Reproducing the Bugs

### Bug 1: _build_path_iterator raises wrong exception types

```python
from Cython.Compiler.TreePath import _build_path_iterator

_build_path_iterator('')

_build_path_iterator('=')

_build_path_iterator('/a')
```

### Bug 2: parse_path_value uses assert for validation

```python
from Cython.Compiler.TreePath import parse_path_value


class MockNext:
    def __init__(self, token):
        self.token = token
        self.called = False

    def __call__(self):
        if self.called:
            raise StopIteration
        self.called = True
        return self.token


next_obj = MockNext()
next_obj.token = ("'abc\"", '')
next_obj.called = False
parse_path_value(next_obj)
```

## Why This Is A Bug

**Issue 1: Inconsistent exception types in `_build_path_iterator`**

The function raises `ValueError` for some invalid inputs (line 281), establishing that malformed paths should produce `ValueError`. However:

1. **Empty path** causes `StopIteration` at line 275 when `_next()` is called on an empty token stream
2. **Unknown operators** like `'='` or `'/'` cause `KeyError` at line 279 when looking up `operations[token[0]]`

This exposes implementation details and makes error handling unpredictable.

**Issue 2: `assert` statements used for validation in `parse_path_value`**

Lines 171 and 174 use `assert` to validate quote matching:
```python
assert value[-1] == value[0]  # line 171
assert value[-1] == value[1]  # line 174
```

This is problematic because:
1. Running Python with `-O` (optimize) disables assertions, allowing invalid input through
2. `AssertionError` is raised instead of `ValueError`, breaking error handling consistency
3. Assertions should be for internal invariants, not input validation

## Fix

### Fix 1: Consistent error handling in `_build_path_iterator`

```diff
--- a/Cython/Compiler/TreePath.py
+++ b/Cython/Compiler/TreePath.py
@@ -272,13 +272,18 @@ def _build_path_iterator(path):
 def _build_path_iterator(path):
     # parse pattern
     _next = _LookAheadTokenizer(path)
-    token = _next()
+    try:
+        token = _next()
+    except StopIteration:
+        raise ValueError("empty path") from None
     selector = []
     while 1:
         try:
-            selector.append(operations[token[0]](_next, token))
+            operation = operations.get(token[0])
+            if operation is None:
+                raise ValueError(f"invalid operator: '{token[0]}'")
+            selector.append(operation(_next, token))
         except StopIteration:
             raise ValueError("invalid path")
```

### Fix 2: Replace assertions with validation in `parse_path_value`

```diff
--- a/Cython/Compiler/TreePath.py
+++ b/Cython/Compiler/TreePath.py
@@ -168,12 +168,16 @@ def parse_path_value(next):
     value = token[0]
     if value:
         if value[:1] == "'" or value[:1] == '"':
-            assert value[-1] == value[0]
+            if value[-1] != value[0]:
+                raise ValueError(f"mismatched quotes in string: {value}")
             return value[1:-1]
         if value[:2] == "b'" or value[:2] == 'b"':
-            assert value[-1] == value[1]
+            if value[-1] != value[1]:
+                raise ValueError(f"mismatched quotes in byte string: {value}")
             return value[2:-1].encode('UTF-8')
```