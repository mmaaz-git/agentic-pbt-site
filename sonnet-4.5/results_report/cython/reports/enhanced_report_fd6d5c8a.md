# Bug Report: Cython.Build.BuildExecutable TypeError When EXE_EXT Is None

**Target**: `Cython.Build.BuildExecutable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The module-level variable `EXE_EXT` is retrieved using `sysconfig.get_config_var('EXE')` without defensive handling, causing TypeError crashes when the value is None during string concatenation operations.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import sysconfig


@given(st.text(min_size=1, max_size=20))
def test_exe_ext_string_concatenation(basename):
    import importlib
    original_get_config_var = sysconfig.get_config_var

    def mock_get_config_var(name):
        if name == 'EXE':
            return None
        return original_get_config_var(name)

    sysconfig.get_config_var = mock_get_config_var

    try:
        if 'Cython.Build.BuildExecutable' in sys.modules:
            del sys.modules['Cython.Build.BuildExecutable']

        import Cython.Build.BuildExecutable

        from Cython.Build.BuildExecutable import EXE_EXT
        result = basename + EXE_EXT

    finally:
        sysconfig.get_config_var = original_get_config_var


if __name__ == '__main__':
    test_exe_ext_string_concatenation()
```

<details>

<summary>
**Failing input**: `basename='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 34, in <module>
    test_exe_ext_string_concatenation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 9, in test_exe_ext_string_concatenation
    def test_exe_ext_string_concatenation(basename):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 27, in test_exe_ext_string_concatenation
    result = basename + EXE_EXT
             ~~~~~~~~~^~~~~~~~~
TypeError: can only concatenate str (not "NoneType") to str
Falsifying example: test_exe_ext_string_concatenation(
    basename='0',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import sysconfig

# Store original function
original_get_config_var = sysconfig.get_config_var

def mock_get_config_var(name):
    if name == 'EXE':
        return None
    return original_get_config_var(name)

# Apply mock
sysconfig.get_config_var = mock_get_config_var

try:
    import importlib
    # Force reimport to use our mocked function
    if 'Cython.Build.BuildExecutable' in sys.modules:
        del sys.modules['Cython.Build.BuildExecutable']

    import Cython.Build.BuildExecutable
    from Cython.Build.BuildExecutable import EXE_EXT, build

    print(f"EXE_EXT value: {EXE_EXT!r}")
    print(f"Type of EXE_EXT: {type(EXE_EXT)}")

    # Try to use it as the module does
    basename = "test_program"
    print(f"\nAttempting string concatenation: basename + EXE_EXT")
    print(f"  basename = {basename!r}")
    print(f"  EXE_EXT = {EXE_EXT!r}")

    # This will fail with TypeError
    exe_name = basename + EXE_EXT
    print(f"Result: {exe_name!r}")

except TypeError as e:
    print(f"\nBUG CONFIRMED - TypeError occurred:")
    print(f"  Error: {e}")
    print(f"  Cannot concatenate string '{basename}' with None")

finally:
    # Restore original function
    sysconfig.get_config_var = original_get_config_var
```

<details>

<summary>
TypeError: can only concatenate str (not "NoneType") to str
</summary>
```
EXE_EXT value: None
Type of EXE_EXT: <class 'NoneType'>

Attempting string concatenation: basename + EXE_EXT
  basename = 'test_program'
  EXE_EXT = None

BUG CONFIRMED - TypeError occurred:
  Error: can only concatenate str (not "NoneType") to str
  Cannot concatenate string 'test_program' with None
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Inconsistent defensive programming**: The module defines a `get_config_var` wrapper function (lines 31-32) that provides default values when config variables are None, but fails to use it for EXE_EXT on line 51.

2. **Python documentation explicitly allows None**: According to Python's official documentation, `sysconfig.get_config_var()` returns None when a variable is not found, making this a documented possibility the code should handle.

3. **Crash prevents all module functionality**: When EXE_EXT is None, both the `build()` function (line 139) and `clink()` function (line 110) crash with unhelpful TypeErrors, completely preventing use of the module rather than gracefully handling the missing configuration.

4. **Pattern violation**: The module already demonstrates awareness of missing config values - other variables like LIBDIR1, LIBDIR2, and CC use the defensive wrapper, but EXE_EXT inexplicably does not.

## Relevant Context

The BuildExecutable module is designed to compile Python scripts into standalone executables. It relies heavily on Python's sysconfig module to obtain platform-specific compilation and linking parameters.

Key observations from the code:
- Lines 31-32 define a `get_config_var` wrapper that returns an empty string default when sysconfig returns None
- Lines 35-50 use this wrapper for most configuration variables (LIBDIR1, LIBDIR2, CC, etc.)
- Line 51 directly calls `sysconfig.get_config_var('EXE')` without the wrapper
- Lines 110 and 139 perform string concatenation with EXE_EXT, assuming it's always a string

The 'EXE' configuration variable typically contains:
- On Windows: '.exe'
- On Unix/Linux: '' (empty string) or potentially undefined

This bug would manifest on platforms where the EXE config variable is not set, in minimal Python installations, or in embedded Python environments.

## Proposed Fix

```diff
--- a/Cython/Build/BuildExecutable.py
+++ b/Cython/Build/BuildExecutable.py
@@ -48,7 +48,7 @@ LINKCC = get_config_var('LINKCC', os.environ.get('LINKCC', CC))
 LINKFORSHARED = get_config_var('LINKFORSHARED')
 LIBS = get_config_var('LIBS')
 SYSLIBS = get_config_var('SYSLIBS')
-EXE_EXT = sysconfig.get_config_var('EXE')
+EXE_EXT = get_config_var('EXE')


 def _debug(msg, *args):
```