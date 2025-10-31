# Bug Report: Cython.Distutils.Extension Loses Cython Parameters When Pyrex Parameters Present

**Target**: `Cython.Distutils.extension.Extension.__init__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When any `pyrex_*` parameter is passed to Extension's constructor alongside explicit `cython_*` parameters, the explicit `cython_*` parameters are silently ignored and reset to their default values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension

valid_identifiers = st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier())

@given(
    name=valid_identifiers,
    cython_value=st.booleans(),
    pyrex_param=st.sampled_from(['pyrex_include_dirs', 'pyrex_directives', 'pyrex_gdb']),
)
def test_cython_params_preserved_with_pyrex(name, cython_value, pyrex_param):
    kwargs = {
        pyrex_param: [] if 'dirs' in pyrex_param or 'directives' in pyrex_param else False,
        'cython_gdb': cython_value,
    }

    ext = Extension(name, [f"{name}.pyx"], **kwargs)

    assert ext.cython_gdb == cython_value, \
        f"cython_gdb should be {cython_value}, but got {ext.cython_gdb} when {pyrex_param} is also present"
```

**Failing input**: `name='A', cython_value=True, pyrex_param='pyrex_include_dirs'`

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension(
    "mymodule",
    ["mymodule.pyx"],
    pyrex_include_dirs=["/legacy/path"],
    cython_gdb=True
)

print(f"cython_gdb = {ext.cython_gdb}")
assert ext.cython_gdb == True
```

Output:
```
cython_gdb = False
AssertionError: cython_gdb should be True
```

## Why This Is A Bug

The backwards compatibility code in `extension.py` lines 40-64 recursively calls `Extension.__init__` when any `pyrex_*` parameter is detected. However, it only passes explicit named parameters and `**kw`, not the cython-specific keyword arguments like `cython_gdb`, `cython_create_listing`, etc.

This means:
1. User passes `pyrex_include_dirs=["/old"]` and `cython_gdb=True`
2. Code detects pyrex option and calls `Extension.__init__(self, ..., no_c_in_traceback=no_c_in_traceback, **kw)`
3. The `cython_gdb` parameter is NOT passed in the recursive call (it's not in kw, and not explicitly listed)
4. In the recursive call, `cython_gdb` gets its default value `False`
5. The explicit `cython_gdb=True` is silently lost

This breaks the reasonable expectation that explicitly-set parameters should be honored, and prevents users from smoothly migrating from pyrex to cython by mixing old and new parameter names.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -45,6 +45,15 @@ class Extension(_Extension.Extension):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
+            # Ensure explicit cython_* parameters are preserved
+            if cython_include_dirs is not None:
+                kw['cython_include_dirs'] = cython_include_dirs
+            if cython_directives is not None:
+                kw['cython_directives'] = cython_directives
+            kw.setdefault('cython_create_listing', cython_create_listing)
+            kw.setdefault('cython_line_directives', cython_line_directives)
+            kw.setdefault('cython_cplus', cython_cplus)
+            kw.setdefault('cython_c_in_temp', cython_c_in_temp)
+            kw.setdefault('cython_gen_pxi', cython_gen_pxi)
+            kw.setdefault('cython_gdb', cython_gdb)
+            kw.setdefault('cython_compile_time_env', cython_compile_time_env)
             Extension.__init__(
                 self, name, sources,
```

This ensures all explicit cython_* parameters are passed through to the recursive call via kw.