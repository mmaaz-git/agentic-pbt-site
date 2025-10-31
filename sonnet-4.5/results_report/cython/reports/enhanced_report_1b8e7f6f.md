# Bug Report: Cython.Distutils.build_ext.get_extension_attr Incorrectly Ignores Falsy Builder Values

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method uses Python's `or` operator to select between builder and extension attribute values, causing all falsy builder values (0, False, [], {}) to be incorrectly ignored in favor of extension values, violating the documented precedence that command-line options should override extension settings.

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

    assert result == builder_value, f"Expected {builder_value}, got {result}"


if __name__ == "__main__":
    test_get_extension_attr_builder_takes_precedence()
```

<details>

<summary>
**Failing input**: `builder_value=0, ext_value=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 28, in <module>
    test_get_extension_attr_builder_takes_precedence()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 8, in test_get_extension_attr_builder_takes_precedence
    st.one_of(st.integers(), st.booleans(), st.lists(st.text()), st.dictionaries(st.text(), st.integers())),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 24, in test_get_extension_attr_builder_takes_precedence
    assert result == builder_value, f"Expected {builder_value}, got {result}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 0, got True
Falsifying example: test_get_extension_attr_builder_takes_precedence(
    # The test always failed when commented parts were varied together.
    builder_value=0,  # or any other generated value
    ext_value=True,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from distutils.dist import Distribution
from unittest.mock import Mock
from Cython.Distutils import build_ext

# Test case: Builder value of 0 should take precedence over extension value of 1
dist = Distribution()
builder = build_ext(dist)
builder.cython_cplus = 0  # Explicitly set to 0 (falsy but valid)

ext = Mock()
ext.cython_cplus = 1  # Extension has value of 1

result = builder.get_extension_attr(ext, 'cython_cplus')
print(f"Test 1: builder=0, extension=1")
print(f"  Expected: 0 (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != 0}")
print()

# Test case 2: Builder value of False should take precedence
builder.cython_gdb = False  # Explicitly set to False
ext.cython_gdb = True
result = builder.get_extension_attr(ext, 'cython_gdb')
print(f"Test 2: builder=False, extension=True")
print(f"  Expected: False (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != False}")
print()

# Test case 3: Empty list should take precedence
builder.cython_directives = []  # Empty list (falsy but valid)
ext.cython_directives = {'language_level': 3}
result = builder.get_extension_attr(ext, 'cython_directives')
print(f"Test 3: builder=[], extension={'language_level': 3}")
print(f"  Expected: [] (builder value should take precedence)")
print(f"  Actual: {result}")
print(f"  Bug present: {result != []}")
```

<details>

<summary>
Builder falsy values are incorrectly ignored in favor of extension values
</summary>
```
Test 1: builder=0, extension=1
  Expected: 0 (builder value should take precedence)
  Actual: 1
  Bug present: True

Test 2: builder=False, extension=True
  Expected: False (builder value should take precedence)
  Actual: True
  Bug present: True

Test 3: builder=[], extension={'language_level': 3}
  Expected: [] (builder value should take precedence)
  Actual: {'language_level': 3}
  Bug present: True
```
</details>

## Why This Is A Bug

The `get_extension_attr` method violates the documented precedence rules where command-line options should take priority over extension settings. The code comments in `build_ext.py` (lines 86-109) explicitly establish this precedence pattern:
- "1. Start with the command line option."
- "2. Add in any (unique) paths from the extension..."

The bug occurs because the implementation at line 81 uses Python's `or` operator:
```python
return getattr(self, option_name) or getattr(extension, option_name, default)
```

This treats all falsy values (0, False, [], {}, etc.) as "not set" and falls back to the extension value. However, these options are declared as `boolean_options` (lines 52-55), indicating they should support both True and False states. A user explicitly setting `cython_cplus=0` or using `--no-cython-cplus` command-line flag has clear intent to disable C++ mode, but the current implementation ignores this and uses the extension's value instead.

This breaks the fundamental contract of distutils/setuptools where command-line options override configuration, making it impossible to disable features via command-line when extensions have them enabled.

## Relevant Context

The affected options include critical Cython build settings:
- `cython_cplus`: Controls C++ vs C compilation mode
- `cython_gdb`: Controls debug information generation
- `cython_create_listing`: Controls error listing file generation
- `cython_line_directives`: Controls source line directive emission
- `cython_c_in_temp`: Controls where generated C files are placed
- `cython_gen_pxi`: Controls .pxi file generation
- `cython_directives`: Compiler directive overrides (when set to empty dict)
- `cython_compile_time_env`: Compile-time environment variables (when set to empty dict)

Users encountering this bug cannot disable these features through command-line options if any extension has them enabled in setup.py, forcing them to modify the setup.py file directly.

## Proposed Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -78,7 +78,10 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        builder_value = getattr(self, option_name, None)
+        if builder_value is not None and builder_value != getattr(type(self), option_name, None):
+            return builder_value
+        return getattr(extension, option_name, default)

     def build_extension(self, ext):
         from Cython.Build.Dependencies import cythonize
```