# Bug Report: pyximport.handle_special_build AttributeError with make_setup_args Only

**Target**: `pyximport.handle_special_build`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `handle_special_build` function crashes with AttributeError when a `.pyxbld` file defines only `make_setup_args()` without `make_ext()`, despite the code explicitly asserting that either function is sufficient.

## Property-Based Test

```python
import os
import tempfile
from hypothesis import given, strategies as st, settings
import pyximport


@given(st.booleans())
@settings(max_examples=50)
def test_handle_special_build_with_setup_args_only(include_make_ext):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        if include_make_ext:
            pyxbld_content = '''
from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''
        else:
            pyxbld_content = '''
def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''

        with open(pyxbld_file, 'w') as f:
            f.write(pyxbld_content)

        ext, setup_args = pyximport.handle_special_build('test', pyxfile)
```

<details>

<summary>
**Failing input**: `include_make_ext=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 40, in <module>
    test_handle_special_build_with_setup_args_only()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 8, in test_handle_special_build_with_setup_args_only
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 36, in test_handle_special_build_with_setup_args_only
    ext, setup_args = pyximport.handle_special_build('test', pyxfile)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py", line 132, in handle_special_build
    for source in ext.sources]
                  ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'sources'
Falsifying example: test_handle_special_build_with_setup_args_only(
    include_make_ext=False,
)
```
</details>

## Reproducing the Bug

```python
import os
import tempfile
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, 'test.pyx')
    pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

    with open(pyxfile, 'w') as f:
        f.write('# cython code')

    with open(pyxbld_file, 'w') as f:
        f.write('''def make_setup_args():
    return {'extra_compile_args': ['-O3']}
''')

    ext, setup_args = pyximport.handle_special_build('test', pyxfile)
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'sources'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/repo.py", line 17, in <module>
    ext, setup_args = pyximport.handle_special_build('test', pyxfile)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py", line 132, in handle_special_build
    for source in ext.sources]
                  ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'sources'
```
</details>

## Why This Is A Bug

The function contains an assertion at line 129 that explicitly states `assert ext or setup_args`, using the logical OR operator to indicate that either `ext` OR `setup_args` is acceptable. This clearly documents the intended behavior that the function should work when only `setup_args` is provided (with `ext` being None).

However, immediately after this assertion, at lines 131-132, the code unconditionally accesses `ext.sources` without checking if `ext` is None. This violates the contract established by the assertion and causes a crash when a `.pyxbld` file defines only `make_setup_args()`.

The error message `'NoneType' object has no attribute 'sources'` is unhelpful compared to what would happen if the assertion required both functions - it would provide a clearer error about missing `make_ext`.

## Relevant Context

The `.pyxbld` file mechanism in pyximport allows users to customize how Cython modules are built without needing a full setup.py file. According to the code documentation (lines 17-28 in pyximport.py), a `.pyxbld` file can define:

- `make_ext(modname, pyxfilename)`: Returns a distutils.extension.Extension object
- `make_setup_args()`: Returns a dictionary of setup arguments for distutils

The code at line 129 explicitly supports having only one of these functions, but the implementation fails to handle the case where only `make_setup_args` is provided.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/pyximport/pyximport.py:110-133`

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