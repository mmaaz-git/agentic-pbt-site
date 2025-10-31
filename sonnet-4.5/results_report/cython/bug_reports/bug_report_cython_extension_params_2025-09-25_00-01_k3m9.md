# Bug Report: Cython.Distutils.Extension Ignores cython_* Parameters When pyrex_* Present

**Target**: `Cython.Distutils.Extension`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When any `pyrex_*` keyword argument is passed to `Extension.__init__`, all explicit `cython_*` constructor parameters (like `cython_include_dirs`, `cython_directives`) are silently ignored and reset to their default values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension


@given(
    st.lists(st.text(min_size=1), min_size=1),
    st.dictionaries(st.text(min_size=1), st.integers()),
)
def test_extension_cython_params_preserved_with_pyrex_kwargs(sources, directives):
    ext = Extension(
        name="test",
        sources=sources,
        cython_directives=directives,
        pyrex_cplus=True,
    )

    assert ext.cython_directives == directives
```

**Failing input**: `sources=['0'], directives={'0': 0}`

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension(
    name="test",
    sources=["test.pyx"],
    cython_include_dirs=["path1", "path2"],
    cython_directives={"language_level": 3},
    pyrex_cplus=True,
)

print(ext.cython_include_dirs)
print(ext.cython_directives)
```

Expected output:
```
['path1', 'path2']
{'language_level': 3}
```

Actual output:
```
[]
{}
```

## Why This Is A Bug

The Extension class provides backward compatibility by accepting both `pyrex_*` and `cython_*` parameters. However, when any `pyrex_*` kwarg is present, the constructor does an early return (line 64 in extension.py) before setting the cython-specific attributes (lines 83-92). This means explicit `cython_include_dirs`, `cython_directives`, and other cython parameters passed to the constructor are completely ignored.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -46,14 +46,22 @@ class Extension(_Extension.Extension):
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
```