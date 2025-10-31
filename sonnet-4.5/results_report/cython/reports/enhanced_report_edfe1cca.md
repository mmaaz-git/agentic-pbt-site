# Bug Report: Cython.Distutils.build_ext Command-Line Directives Cause Crash Due to Missing String Parsing

**Target**: `Cython.Distutils.build_ext.finalize_options`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `--cython-directives` command-line option crashes when used because `finalize_options()` doesn't parse string values to dictionaries, causing a ValueError when `build_extension()` attempts to convert the string with `dict()`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython.Distutils.build_ext directives type validation"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from distutils.dist import Distribution
from Cython.Distutils import build_ext

@settings(max_examples=100)
@given(st.text(min_size=1, max_size=100))
def test_directives_type_validation(directive_value):
    """
    Property: finalize_options should ensure cython_directives is a dict
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    cmd.cython_directives = directive_value
    cmd.finalize_options()

    assert isinstance(cmd.cython_directives, dict), \
        f"Expected dict, got {type(cmd.cython_directives)}"

# Run the test
if __name__ == "__main__":
    print("Running property-based test for cython_directives type validation...")
    print("=" * 70)
    try:
        test_directives_type_validation()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nNote: This demonstrates that finalize_options() does not")
        print("convert string values to dict as required by build_extension()")
```

<details>

<summary>
**Failing input**: Any string value (e.g., "boundscheck=True")
</summary>
```
Running property-based test for cython_directives type validation...
======================================================================
Test failed: Expected dict, got <class 'str'>

Note: This demonstrates that finalize_options() does not
convert string values to dict as required by build_extension()
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython.Distutils.build_ext crash with string directives"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import build_ext

# Create a distribution and build_ext command
dist = Distribution()
cmd = build_ext(dist)
cmd.initialize_options()

# Set cython_directives as a string (as would happen from command line)
cmd.cython_directives = "boundscheck=True,wraparound=False"
print(f"Before finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# Call finalize_options (should parse string to dict, but doesn't)
cmd.finalize_options()
print(f"\nAfter finalize_options: cython_directives = {repr(cmd.cython_directives)}")
print(f"Type: {type(cmd.cython_directives)}")

# This is what build_extension does on line 107, which will crash
print("\nAttempting dict(cmd.cython_directives) as done in build_extension()...")
try:
    directives = dict(cmd.cython_directives)
    print(f"Success: directives = {directives}")
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError when attempting to convert string to dict
</summary>
```
Before finalize_options: cython_directives = 'boundscheck=True,wraparound=False'
Type: <class 'str'>

After finalize_options: cython_directives = 'boundscheck=True,wraparound=False'
Type: <class 'str'>

Attempting dict(cmd.cython_directives) as done in build_extension()...
CRASH: ValueError: dictionary update sequence element #0 has length 1; 2 is required
```
</details>

## Why This Is A Bug

This is a legitimate bug because the `--cython-directives` option is explicitly defined as a command-line argument in the `user_options` list (lines 44-45 of build_ext.py), yet it crashes when actually used from the command line. The implementation violates several expected behaviors:

1. **Incomplete type handling**: The `finalize_options()` method (lines 77-78) only handles the `None` case, converting it to an empty dict, but fails to handle string values that come from command-line usage.

2. **Inconsistent with similar options**: The parallel option `--cython-include-dirs` IS properly parsed from string to list in `finalize_options()` (lines 74-76), showing that string parsing is the expected pattern.

3. **Documented interface failure**: The option appears in `user_options`, making it part of the public API that users expect to work via `python setup.py build_ext --cython-directives="..."`.

4. **Assumption mismatch**: The `build_extension()` method (line 107) assumes `cython_directives` is already a dictionary and calls `dict(self.cython_directives)` which crashes when given a string.

5. **Confusing error message**: The resulting ValueError ("dictionary update sequence element #0 has length 1; 2 is required") doesn't indicate the actual problem, making debugging difficult for users.

## Relevant Context

The bug manifests when users try to override Cython compiler directives from the command line, which is useful in CI/CD pipelines or when testing different optimization settings without modifying setup.py files.

Key code locations:
- Definition of option: `/Cython/Distutils/build_ext.py:44-45`
- Initialization: `/Cython/Distutils/build_ext.py:63`
- Incomplete parsing: `/Cython/Distutils/build_ext.py:77-78`
- Crash location: `/Cython/Distutils/build_ext.py:107`

The expected command-line usage would be:
```bash
python setup.py build_ext --cython-directives="boundscheck=False,wraparound=False,language_level=3"
```

Current workaround requires modifying setup.py:
```python
from Cython.Distutils import build_ext
# ... in setup()
ext_modules = cythonize(extensions, compiler_directives={'boundscheck': False})
```

## Proposed Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -76,6 +76,17 @@ class build_ext(_build_ext):
                 self.cython_include_dirs.split(os.pathsep)
         if self.cython_directives is None:
             self.cython_directives = {}
+        elif isinstance(self.cython_directives, str):
+            directives = {}
+            if self.cython_directives.strip():
+                for directive in self.cython_directives.split(','):
+                    directive = directive.strip()
+                    if '=' in directive:
+                        key, value = directive.split('=', 1)
+                        directives[key.strip()] = value.strip()
+                    elif directive:
+                        directives[directive] = True
+            self.cython_directives = directives

     def get_extension_attr(self, extension, option_name, default=False):
         return getattr(self, option_name) or getattr(extension, option_name, default)
```