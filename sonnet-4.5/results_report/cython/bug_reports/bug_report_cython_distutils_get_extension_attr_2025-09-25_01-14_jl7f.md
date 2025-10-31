# Bug Report: Cython.Distutils.build_ext.get_extension_attr Ignores Falsy Command-Line Values

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method uses `or` logic to combine command-line and extension attributes, causing it to ignore explicitly-set falsy command-line values (0, False, [], "") and incorrectly fall back to extension values instead.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

@given(
    cmd_value=st.one_of(
        st.just(0),
        st.just(False),
        st.just([]),
        st.just(""),
    ),
    ext_value=st.one_of(
        st.just(1),
        st.just(True),
        st.just(["/some/path"]),
        st.just("something"),
    ),
)
def test_get_extension_attr_falsy_command_values(cmd_value, ext_value):
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()
    cmd.finalize_options()

    cmd.cython_gdb = cmd_value

    ext = Extension("test_module", ["test.pyx"])
    ext.cython_gdb = ext_value

    result = cmd.get_extension_attr(ext, 'cython_gdb')

    assert result == cmd_value, \
        f"Expected get_extension_attr to return command value {cmd_value!r}, but got {result!r}"
```

**Failing input**: `cmd_value=0, ext_value=1` (or any falsy cmd_value with truthy ext_value)

## Reproducing the Bug

```python
from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()
cmd.finalize_options()

cmd.cython_gdb = False

ext = Extension("test_module", ["test.pyx"])
ext.cython_gdb = True

result = cmd.get_extension_attr(ext, 'cython_gdb')

print(f"Command line: cython_gdb = {cmd.cython_gdb}")
print(f"Extension: cython_gdb = {ext.cython_gdb}")
print(f"Result: {result}")

assert result == False, f"Expected False (from command), got {result}"
```

Output:
```
Command line: cython_gdb = False
Extension: cython_gdb = True
Result: True
AssertionError: Expected False (from command), got True
```

## Why This Is A Bug

The method's purpose is to prioritize command-line options over extension-level settings. However, the current implementation uses the `or` operator, which treats all falsy values (0, False, [], "") as "not set", causing the method to incorrectly fall back to the extension value.

This prevents users from disabling features via command-line options when the extension has them enabled. For example:
- Setting `--cython-gdb=False` to disable debugging
- Setting `--cython-cplus=0` to disable C++ mode
- Setting `--cython-gen-pxi=False` to disable .pxi generation

All of these explicitly-set command-line values would be ignored.

## Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -83,7 +83,11 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        cmd_value = getattr(self, option_name, None)
+        if cmd_value is not None:
+            return cmd_value
+        else:
+            return getattr(extension, option_name, default)
```

This fix checks if the command attribute exists and is not None (rather than checking truthiness), ensuring that explicitly-set falsy values are respected.