# Bug Report: Cython.Distutils.Extension Parameter Loss with Mixed pyrex/cython Options

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When constructing an Extension with explicit `cython_*` parameters (e.g., `cython_include_dirs`) alongside any `pyrex_*` keyword arguments (e.g., `pyrex_gdb=True`), the explicit `cython_*` parameters are silently lost and reset to their default values (empty list/dict).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension


@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3),
    st.booleans(),
)
def test_extension_explicit_cython_with_pyrex_kwarg(include_dirs, pyrex_gdb):
    ext = Extension(
        "test",
        ["test.pyx"],
        cython_include_dirs=include_dirs,
        pyrex_gdb=pyrex_gdb,
    )

    assert ext.cython_include_dirs == include_dirs
    assert ext.cython_gdb == pyrex_gdb
```

**Failing input**: `include_dirs=['0'], pyrex_gdb=False`

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension(
    "test",
    ["test.pyx"],
    cython_include_dirs=['/my/include/path'],
    pyrex_gdb=True,
)

print(f"Expected: ['/my/include/path']")
print(f"Actual: {ext.cython_include_dirs}")

ext2 = Extension(
    "test2",
    ["test2.pyx"],
    cython_directives={'boundscheck': False},
    pyrex_cplus=True,
)

print(f"Expected: {{'boundscheck': False}}")
print(f"Actual: {ext2.cython_directives}")
```

Output:
```
Expected: ['/my/include/path']
Actual: []
Expected: {'boundscheck': False}
Actual: {}
```

## Why This Is A Bug

The Extension class's pyrex → cython translation logic (lines 40-64 in extension.py) makes a recursive call when any `pyrex_*` keyword argument is detected. However, this recursive call only passes distutils base class parameters and `no_c_in_traceback`, omitting all other explicit `cython_*` parameters (lines 47-63). As a result, users who mix the modern `cython_*` parameter syntax with deprecated `pyrex_*` kwargs lose their configuration silently, likely causing build failures or incorrect compilation settings.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -44,6 +44,14 @@ class Extension(_Extension.Extension):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
+            # Preserve explicit cython_* parameters by adding them to kw if not already present
+            for param_name in ['cython_include_dirs', 'cython_directives', 'cython_create_listing',
+                               'cython_line_directives', 'cython_cplus', 'cython_c_in_temp',
+                               'cython_gen_pxi', 'cython_gdb', 'cython_compile_time_env']:
+                param_value = locals()[param_name]
+                if param_value is not None and param_name not in kw:
+                    kw[param_name] = param_value
+
             Extension.__init__(
                 self, name, sources,
                 include_dirs=include_dirs,
```