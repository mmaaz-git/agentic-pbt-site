# Bug Report: Cython.Distutils.build_ext.get_extension_attr Ignores Falsy Command-Line Options

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method in `Cython.Distutils.build_ext` uses the `or` operator to choose between command-line and extension-level settings. This causes falsy but valid values (like `0`, `False`, `""`, `[]`, `{}`) from command-line options to be incorrectly ignored in favor of extension-level settings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


@given(
    st.one_of(st.just(0), st.just(False), st.just(""), st.just([]))
)
@settings(max_examples=100)
def test_get_extension_attr_falsy_bug(falsy_value):
    dist = Distribution()
    build_ext_instance = build_ext(dist)
    build_ext_instance.initialize_options()
    build_ext_instance.finalize_options()

    class MockExtension:
        pass

    ext = MockExtension()

    setattr(build_ext_instance, 'test_option', falsy_value)
    setattr(ext, 'test_option', "extension_value")

    result = build_ext_instance.get_extension_attr(ext, 'test_option', default="default")

    assert result == falsy_value
```

**Failing input**: `falsy_value=0` (and also `False`, `""`, `[]`, `{}`)

## Reproducing the Bug

Example 1: Boolean option with value 0

```python
from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


class MockExtension:
    cython_cplus = 1


dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

build_ext_instance.cython_cplus = 0

ext = MockExtension()

result = build_ext_instance.get_extension_attr(ext, 'cython_cplus')

print(f"Expected: 0, Actual: {result}")
```

Example 2: Compile-time environment with empty dict

```python
from Cython.Distutils.build_ext import build_ext
from Cython.Distutils import Extension
from distutils.dist import Distribution

dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

build_ext_instance.cython_compile_time_env = {}

ext = Extension(
    "test_module",
    ["test.pyx"],
    cython_compile_time_env={"DEBUG": True}
)

result = build_ext_instance.get_extension_attr(ext, 'cython_compile_time_env', default=None)

print(f"Expected: {{}}, Actual: {result}")
```

## Why This Is A Bug

The method is designed to prioritize command-line settings (from `self`) over extension-level settings. However, the current implementation uses:

```python
return getattr(self, option_name) or getattr(extension, option_name, default)
```

This incorrectly treats falsy values like `0`, `False`, `""`, `[]`, or `{}` as "not set" and falls through to the extension's value. Real-world impacts:

1. **Boolean options**: If a user sets a boolean flag to `0` at the command level, but an extension has it set to `1`, the command-line setting is ignored. For example, `cython_cplus = 0` is overridden by extension's `cython_cplus = True`.

2. **Empty collections**: If a user explicitly sets `cython_compile_time_env = {}` to clear compile-time environment variables, but an extension has values, the command-line empty dict is ignored and the extension's values are used instead.

This violates the principle that command-line options should override extension settings.

## Fix

The fix needs to handle the distinction between "not set" and "set to a falsy value". Looking at the codebase:

- Some options (like `cython_compile_time_env`) use `None` as the sentinel for "not set"
- Other options (like `cython_directives`) are converted from `None` to `{}` in `finalize_options`
- Boolean options use `0` for "not set"

The most robust fix is to check if the self value is the initialized default (which for most non-boolean options is `None`):

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -78,7 +78,12 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        self_value = getattr(self, option_name, None)
+        ext_value = getattr(extension, option_name, default)
+
+        if self_value is not None and self_value != getattr(self.__class__, option_name, None):
+            return self_value
+        return ext_value if ext_value is not None else self_value
```

Alternatively, a simpler fix that works for the common cases:

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -78,7 +78,8 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        self_value = getattr(self, option_name, None)
+        return self_value if self_value is not None else getattr(extension, option_name, default)
```

The simpler fix relies on `None` being the standard sentinel for "not set", which works for most options but may have edge cases with boolean options that initialize to `0` instead of `None`.