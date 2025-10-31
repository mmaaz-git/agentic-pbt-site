# Bug Report: Cython.Distutils.Extension Parameter Loss with Pyrex Options

**Target**: `Cython.Distutils.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When any `pyrex_*` parameter is used with `Extension.__init__`, all named `cython_*` parameters are silently lost due to a recursive call that doesn't forward them.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension

@given(
    st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier()),
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3),
    st.dictionaries(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        st.booleans(),
        min_size=1, max_size=3
    )
)
def test_extension_init_preserves_cython_params_with_pyrex(module_name, include_dirs, directives):
    ext = Extension(
        module_name,
        [f"{module_name}.pyx"],
        pyrex_gdb=True,
        cython_include_dirs=include_dirs,
        cython_directives=directives
    )

    assert ext.cython_include_dirs == include_dirs
    assert ext.cython_directives == directives
```

**Failing input**: `module_name='A'`, `include_dirs=['0']`, `directives={'A': False}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Distutils import Extension

ext = Extension(
    "mymodule",
    ["mymodule.pyx"],
    pyrex_gdb=True,
    cython_include_dirs=["/usr/local/include"],
    cython_directives={"boundscheck": False}
)

print(f"cython_include_dirs: {ext.cython_include_dirs}")
print(f"cython_directives: {ext.cython_directives}")
print(f"cython_gdb: {ext.cython_gdb}")
```

Output:
```
cython_include_dirs: []
cython_directives: {}
cython_gdb: True
```

## Why This Is A Bug

In `extension.py` lines 40-64, when any `pyrex_*` option is provided via `**kw`, the code:
1. Converts `pyrex_X` to `cython_X` in the `kw` dict
2. Recursively calls `Extension.__init__` with `**kw`
3. Returns early, skipping lines 83-92 that set `self.cython_*` attributes

The recursive call at lines 47-64 passes all the standard distutils parameters and `**kw`, but it does NOT pass the named `cython_*` parameters (`cython_include_dirs`, `cython_directives`, etc.). This causes these parameters to use their default values (empty list/dict) instead of the user-provided values.

This is a silent data loss bug - users' configuration is ignored without any warning.

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -44,6 +44,15 @@ class Extension(_Extension.Extension):
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