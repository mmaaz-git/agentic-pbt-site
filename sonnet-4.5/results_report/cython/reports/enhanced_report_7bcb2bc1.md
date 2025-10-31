# Bug Report: Cython.Distutils.build_ext.get_extension_attr Ignores Explicitly-Set Falsy Command-Line Values

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method incorrectly ignores explicitly-set falsy command-line values (0, False, [], "") due to its use of the `or` operator, causing it to fall back to extension values when it should prioritize the command-line settings.

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

# Run the test
test_get_extension_attr_falsy_command_values()
```

<details>

<summary>
**Failing input**: `cmd_value=0, ext_value=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 36, in <module>
    test_get_extension_attr_falsy_command_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 6, in test_get_extension_attr_falsy_command_values
    cmd_value=st.one_of(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 32, in test_get_extension_attr_falsy_command_values
    assert result == cmd_value, \
           ^^^^^^^^^^^^^^^^^^^
AssertionError: Expected get_extension_attr to return command value 0, but got 1
Falsifying example: test_get_extension_attr_falsy_command_values(
    cmd_value=0,  # or any other generated value
    ext_value=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from distutils.dist import Distribution
from Cython.Distutils import build_ext, Extension

dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()
cmd.finalize_options()

# Set command line option to False (explicitly disabling)
cmd.cython_gdb = False

ext = Extension("test_module", ["test.pyx"])
# Extension has this enabled
ext.cython_gdb = True

result = cmd.get_extension_attr(ext, 'cython_gdb')

print(f"Command line: cython_gdb = {cmd.cython_gdb}")
print(f"Extension: cython_gdb = {ext.cython_gdb}")
print(f"Result: {result}")
print(f"Expected: {cmd.cython_gdb} (from command line)")
print()

if result != cmd.cython_gdb:
    print(f"ERROR: Expected {cmd.cython_gdb} (from command), got {result}")
    print("This is a bug - command line value should take precedence!")
else:
    print("SUCCESS: Command line value correctly takes precedence")
```

<details>

<summary>
ERROR: Command line value False is incorrectly overridden by extension value True
</summary>
```
Command line: cython_gdb = False
Extension: cython_gdb = True
Result: True
Expected: False (from command line)

ERROR: Expected False (from command), got True
This is a bug - command line value should take precedence!
```
</details>

## Why This Is A Bug

This violates the fundamental principle of command-line tools where command-line options should override configuration-level settings. The `get_extension_attr` method is specifically designed to prioritize command-line values over extension-level values, as evidenced by its usage throughout the `build_extension` method (lines 111-128 in build_ext.py).

The current implementation on line 81 uses `return getattr(self, option_name) or getattr(extension, option_name, default)`, which treats all falsy values as "not set". This means:
- Setting `--cython-gdb=False` to disable debugging won't work if the extension has `cython_gdb=True`
- Setting `--cython-cplus=0` to disable C++ mode won't work if the extension has `cython_cplus=1`
- Any boolean option set to `False` or numeric option set to `0` via command-line gets ignored

This breaks the expected precedence order and prevents users from disabling features at build time via command-line flags.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/build_ext.py` at lines 80-81.

The method is used extensively throughout the build process to determine option values for:
- `cython_cplus` (C++ mode) - line 111
- `cython_create_listing` (listing file generation) - line 119
- `cython_line_directives` (line number directives) - line 120
- `cython_c_in_temp` (temporary C file location) - line 123
- `cython_gen_pxi` (.pxi file generation) - line 124
- `cython_gdb` (GDB debug info) - line 125
- `cython_compile_time_env` (compile-time environment) - line 127

All of these options can be affected when trying to explicitly disable them via command-line.

## Proposed Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -78,7 +78,11 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        cmd_value = getattr(self, option_name, None)
+        if cmd_value is not None:
+            return cmd_value
+        else:
+            return getattr(extension, option_name, default)
```