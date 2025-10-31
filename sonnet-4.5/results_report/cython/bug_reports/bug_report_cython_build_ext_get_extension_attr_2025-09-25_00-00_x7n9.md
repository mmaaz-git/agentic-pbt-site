# Bug Report: build_ext.get_extension_attr Ignores Falsy Builder Values

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method uses `or` operator to select between builder and extension attribute values, causing falsy builder values (0, False, [], {}) to be incorrectly ignored in favor of extension values, even though builder options should take precedence.

## Property-Based Test

```python
from distutils.dist import Distribution
from unittest.mock import Mock
from hypothesis import given, assume, strategies as st
from Cython.Distutils import build_ext


@given(
    st.one_of(st.integers(), st.booleans(), st.lists(st.text()), st.dictionaries(st.text(), st.integers())),
    st.one_of(st.integers(), st.booleans(), st.lists(st.text()), st.dictionaries(st.text(), st.integers())),
)
def test_get_extension_attr_builder_takes_precedence(builder_value, ext_value):
    assume(builder_value != ext_value)
    assume(not builder_value and ext_value)

    dist = Distribution()
    builder = build_ext(dist)
    builder.cython_cplus = builder_value

    ext = Mock()
    ext.cython_cplus = ext_value

    result = builder.get_extension_attr(ext, 'cython_cplus')

    assert result == builder_value
```

**Failing input**: `builder_value=0, ext_value=1`

## Reproducing the Bug

```python
from distutils.dist import Distribution
from unittest.mock import Mock
from Cython.Distutils import build_ext

dist = Distribution()
builder = build_ext(dist)
builder.cython_cplus = 0

ext = Mock()
ext.cython_cplus = 1

result = builder.get_extension_attr(ext, 'cython_cplus')
print(f"Result: {result}")
```

Expected output: `Result: 0`
Actual output: `Result: 1`

## Why This Is A Bug

The `get_extension_attr` method is designed to merge builder (command-line) options with extension-specific options, with builder options taking precedence. However, the implementation uses `or` operator:

```python
return getattr(self, option_name) or getattr(extension, option_name, default)
```

This treats falsy values (0, False, [], {}) as "not set", causing them to fall back to the extension's value. This violates the precedence rule documented in the code comments (lines 86-101 of build_ext.py) that state command-line options should take priority.

A user who explicitly sets `--no-cython-cplus` or `cython_cplus=0` expects this to disable C++ mode, but if an extension has `cython_cplus=1`, the extension's value incorrectly wins.

## Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -79,7 +79,10 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        self_value = getattr(self, option_name, None)
+        if self_value is not None and self_value != getattr(type(self), option_name, None):
+            return self_value
+        return getattr(extension, option_name, default)

     def build_extension(self, ext):
         from Cython.Build.Dependencies import cythonize
```