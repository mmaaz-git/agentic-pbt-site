# Bug Report: Cython.Distutils pyrex_* Silently Overwrites cython_*

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When both deprecated `pyrex_*` and modern `cython_*` parameters are provided to the Extension constructor, the `pyrex_*` value silently overwrites the `cython_*` value without warning or error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@settings(max_examples=200)
@given(
    st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3)
)
def test_pyrex_cython_conflict_behavior(module_name, pyrex_dirs, cython_dirs):
    """
    Property: When both pyrex_* and cython_* are provided, behavior should be
    well-defined (either raise error or have documented precedence)
    """
    ext = Extension(
        module_name,
        [f"{module_name}.pyx"],
        pyrex_include_dirs=pyrex_dirs,
        cython_include_dirs=cython_dirs
    )

    result = ext.cython_include_dirs

    if result == pyrex_dirs and pyrex_dirs != cython_dirs:
        raise AssertionError("pyrex_* silently overwrites cython_* when both provided")
```

**Failing input**: `module_name='A', pyrex_dirs=['0'], cython_dirs=['00']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Distutils import Extension

ext = Extension(
    "test_module",
    ["test.pyx"],
    pyrex_include_dirs=["/pyrex/path"],
    cython_include_dirs=["/cython/path"]
)

print(f"cython_include_dirs: {ext.cython_include_dirs}")
```

**Output**: `cython_include_dirs: ['/pyrex/path']`
**Expected**: Either `['/cython/path']` or a clear error message

## Why This Is A Bug

The Extension constructor processes `pyrex_*` parameters for backward compatibility by renaming them to `cython_*` in the keyword arguments dictionary (lines 42-45 in extension.py). However, when both `pyrex_include_dirs` and `cython_include_dirs` are explicitly provided:

1. The code renames `pyrex_include_dirs` to `cython_include_dirs` in the `kw` dict
2. This overwrites the existing `cython_include_dirs` value that was also in `kw`
3. No warning or error is raised
4. The modern `cython_*` parameter is silently discarded

This is problematic because:
- Users migrating from `pyrex_*` to `cython_*` might accidentally provide both during transition
- The modern parameter should take precedence over the deprecated one
- Silent data loss violates the principle of least surprise
- No documentation warns about this behavior

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -40,8 +40,15 @@ class Extension(_Extension.Extension):
         # Translate pyrex_X to cython_X for backwards compatibility.
         had_pyrex_options = False
         for key in list(kw):
             if key.startswith('pyrex_'):
                 had_pyrex_options = True
-                kw['cython' + key[5:]] = kw.pop(key)
+                cython_key = 'cython' + key[5:]
+                if cython_key in kw:
+                    import warnings
+                    warnings.warn(
+                        f"Both {key} and {cython_key} provided. Using {cython_key}.",
+                        DeprecationWarning, stacklevel=2
+                    )
+                    kw.pop(key)
+                else:
+                    kw[cython_key] = kw.pop(key)
         if had_pyrex_options:
```