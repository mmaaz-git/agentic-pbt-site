# Bug Report: pyximport.handle_special_build AttributeError with make_setup_args only

**Target**: `pyximport.handle_special_build`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `handle_special_build()` function crashes with AttributeError when a `.pyxbld` file defines only `make_setup_args()` without `make_ext()`, despite the code explicitly allowing this configuration through its assertion logic.

## Property-Based Test

```python
import os
import tempfile
import sys
from hypothesis import given, strategies as st

# Add the pyximport to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
import pyximport

@given(st.just(None))
def test_handle_special_build_with_only_setup_args(x):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_file = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyx_file, 'w') as f:
            f.write('def hello(): return "world"')

        with open(pyxbld_file, 'w') as f:
            f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

        ext, setup_args = pyximport.handle_special_build('test', pyx_file)
        assert isinstance(setup_args, dict)

if __name__ == "__main__":
    test_handle_special_build_with_only_setup_args()
```

<details>

<summary>
**Failing input**: `x=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 26, in <module>
    test_handle_special_build_with_only_setup_args()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 11, in test_handle_special_build_with_only_setup_args
    def test_handle_special_build_with_only_setup_args(x):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 22, in test_handle_special_build_with_only_setup_args
    ext, setup_args = pyximport.handle_special_build('test', pyx_file)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py", line 132, in handle_special_build
    for source in ext.sources]
                  ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'sources'
Falsifying example: test_handle_special_build_with_only_setup_args(
    x=None,
)
```
</details>

## Reproducing the Bug

```python
import os
import tempfile
import sys

# Add the pyximport to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyx_file = os.path.join(tmpdir, 'example.pyx')
    pyxbld_file = os.path.join(tmpdir, 'example.pyxbld')

    with open(pyx_file, 'w') as f:
        f.write('def hello(): return "world"')

    with open(pyxbld_file, 'w') as f:
        f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

    ext, setup_args = pyximport.handle_special_build('example', pyx_file)
    print(f"Extension: {ext}")
    print(f"Setup args: {setup_args}")
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'sources'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/repo.py", line 19, in <module>
    ext, setup_args = pyximport.handle_special_build('example', pyx_file)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py", line 132, in handle_special_build
    for source in ext.sources]
                  ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'sources'
```
</details>

## Why This Is A Bug

The function `handle_special_build` in pyximport.py has a logical error that contradicts its design intent. The assertion at lines 129-130 explicitly allows for either `ext` OR `setup_args` to exist:

```python
assert ext or setup_args, ("neither make_ext nor make_setup_args %s" % special_build)
```

This assertion passes when `ext=None` and `setup_args` contains a dictionary, which is exactly what happens when a .pyxbld file defines only `make_setup_args()`. However, immediately after this assertion, lines 131-132 unconditionally access `ext.sources`:

```python
ext.sources = [os.path.join(os.path.dirname(special_build), source)
               for source in ext.sources]
```

This causes an AttributeError because `ext` is None when only `make_setup_args` is defined. The module docstring (lines 17-28) shows an example with both functions but doesn't explicitly require both. The code structure with optional getattr calls (lines 120-128) and the "or" assertion clearly indicates that having only one function should be supported.

## Relevant Context

The pyximport module allows users to customize the build process for .pyx files through .pyxbld files. These files can define:
- `make_ext(modname, pyxfilename)`: Returns a custom distutils Extension object
- `make_setup_args()`: Returns a dictionary of setup arguments for the Distribution

The bug prevents users from using just `make_setup_args()` to configure compiler flags or other build settings without needing to define a custom Extension. This is a reasonable use case, especially when users only want to pass flags like `--compiler=mingw32` as shown in the documentation example.

Code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/pyximport/pyximport.py:110-133`

## Proposed Fix

```diff
--- a/pyximport/pyximport.py
+++ b/pyximport/pyximport.py
@@ -128,8 +128,9 @@ def handle_special_build(modname, pyxfilename):
                                          % special_build)
         assert ext or setup_args, ("neither make_ext nor make_setup_args %s"
                                          % special_build)
-        ext.sources = [os.path.join(os.path.dirname(special_build), source)
-                       for source in ext.sources]
+        if ext:
+            ext.sources = [os.path.join(os.path.dirname(special_build), source)
+                           for source in ext.sources]
     return ext, setup_args
```