# Bug Report: Cython.Distutils.Extension Pyrex/Cython Option Mixing

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When mixing deprecated `pyrex_*` options with `cython_*` options in Extension constructor, all directly specified `cython_*` parameters are silently ignored due to early return in backwards compatibility code path.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Distutils import Extension

@given(st.booleans(), st.booleans())
@settings(max_examples=200)
def test_mixing_pyrex_and_cython_options(pyrex_val, cython_val):
    ext = Extension("test", ["test.pyx"], pyrex_cplus=pyrex_val, cython_gdb=cython_val)

    assert ext.cython_cplus == pyrex_val
    assert ext.cython_gdb == cython_val
```

**Failing input**: `pyrex_val=False, cython_val=True` (pyrex option triggers backwards compatibility path, cython option gets ignored)

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension("test", ["test.pyx"], pyrex_cplus=False, cython_gdb=True)

print(f"Expected cython_gdb: True")
print(f"Actual cython_gdb: {ext.cython_gdb}")
```

Output:
```
Expected cython_gdb: True
Actual cython_gdb: False
```

## Why This Is A Bug

The backwards compatibility code (extension.py:40-64) detects any `pyrex_*` options, translates them to `cython_*` in the `**kw` dict, then recursively calls `__init__` and returns early (line 64). However, the recursive call only passes the explicitly named parameters (`include_dirs`, `language`, `no_c_in_traceback`, etc.) plus `**kw`.

All `cython_*` named parameters (`cython_gdb`, `cython_include_dirs`, `cython_directives`, etc.) are **not passed** to the recursive call, so they revert to their default values. This silently ignores user-specified cython options when any pyrex option is present.

## Fix

```diff
--- a/extension.py
+++ b/extension.py
@@ -40,6 +40,16 @@ class Extension(_Extension.Extension):
         had_pyrex_options = False
         for key in list(kw):
             if key.startswith('pyrex_'):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
+            # Preserve explicitly passed cython_* parameters
+            if cython_include_dirs is not None:
+                kw['cython_include_dirs'] = cython_include_dirs
+            if cython_directives is not None:
+                kw['cython_directives'] = cython_directives
+            if cython_create_listing:
+                kw['cython_create_listing'] = cython_create_listing
+            if cython_line_directives:
+                kw['cython_line_directives'] = cython_line_directives
+            if cython_cplus:
+                kw['cython_cplus'] = cython_cplus
+            if cython_c_in_temp:
+                kw['cython_c_in_temp'] = cython_c_in_temp
+            if cython_gen_pxi:
+                kw['cython_gen_pxi'] = cython_gen_pxi
+            if cython_gdb:
+                kw['cython_gdb'] = cython_gdb
+            if cython_compile_time_env is not None:
+                kw['cython_compile_time_env'] = cython_compile_time_env
             Extension.__init__(
                 self, name, sources,
                 include_dirs=include_dirs,
                 define_macros=define_macros,
                 undef_macros=undef_macros,
                 library_dirs=library_dirs,
                 libraries=libraries,
                 runtime_library_dirs=runtime_library_dirs,
                 extra_objects=extra_objects,
                 extra_compile_args=extra_compile_args,
                 extra_link_args=extra_link_args,
                 export_symbols=export_symbols,
                 depends=depends,
                 language=language,
                 no_c_in_traceback=no_c_in_traceback,
                 **kw)
             return
```