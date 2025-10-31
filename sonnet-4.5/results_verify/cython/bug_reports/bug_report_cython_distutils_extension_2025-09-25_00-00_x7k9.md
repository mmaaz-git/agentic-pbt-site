# Bug Report: Cython.Distutils.Extension - Mixed pyrex/cython Parameters Lost

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `Extension` is initialized with any `pyrex_*` parameter alongside `cython_*` parameters, all `cython_*` parameters are silently dropped due to incomplete parameter forwarding in the recursive initialization path.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from Cython.Distutils import Extension

@given(
    st.lists(st.text(min_size=1), min_size=1),
    st.dictionaries(st.text(min_size=1), st.text()),
)
@settings(max_examples=500)
def test_extension_mixed_pyrex_cython_parameters(include_dirs, directives):
    ext = Extension(
        "test.module",
        ["test.pyx"],
        pyrex_include_dirs=include_dirs,
        cython_directives=directives,
    )

    assert ext.cython_include_dirs == include_dirs
    assert ext.cython_directives == directives
```

**Failing input**: `include_dirs=['0'], directives={'0': ''}`

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension(
    "test.module",
    ["test.pyx"],
    pyrex_include_dirs=['dir1'],
    cython_directives={'language_level': '3'},
)

assert ext.cython_include_dirs == ['dir1']
assert ext.cython_directives == {}

print(f"BUG: cython_directives is {ext.cython_directives}")
print(f"Expected: {{'language_level': '3'}}")
```

## Why This Is A Bug

The Extension class provides backward compatibility for Pyrex parameters by translating `pyrex_*` to `cython_*`. When any `pyrex_*` parameter is detected, the code recursively calls `Extension.__init__` with translated parameters (extension.py:47-63). However, this recursive call only forwards the base distutils parameters and `**kw`, omitting all Cython-specific named parameters like `cython_directives`, `cython_include_dirs`, `cython_create_listing`, etc. This breaks the expected behavior that mixing old and new parameter names should work during migration.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -44,20 +44,33 @@ class Extension(_Extension.Extension):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
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
                 #swig_opts=swig_opts,
                 depends=depends,
                 language=language,
+                cython_include_dirs=cython_include_dirs,
+                cython_directives=cython_directives,
+                cython_create_listing=cython_create_listing,
+                cython_line_directives=cython_line_directives,
+                cython_cplus=cython_cplus,
+                cython_c_in_temp=cython_c_in_temp,
+                cython_gen_pxi=cython_gen_pxi,
+                cython_gdb=cython_gdb,
                 no_c_in_traceback=no_c_in_traceback,
+                cython_compile_time_env=cython_compile_time_env,
                 **kw)
             return