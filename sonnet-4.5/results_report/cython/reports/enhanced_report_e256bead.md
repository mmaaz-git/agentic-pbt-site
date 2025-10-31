# Bug Report: Cython.Distutils.Extension Silently Ignores cython_* Parameters When pyrex_* Arguments Present

**Target**: `Cython.Distutils.Extension`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When any `pyrex_*` keyword argument is passed to the Extension constructor, all explicitly provided `cython_*` parameters are silently discarded and reset to their default empty values due to a missing parameter pass-through in the recursive constructor call.

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


if __name__ == "__main__":
    test_extension_cython_params_preserved_with_pyrex_kwargs()
```

<details>

<summary>
**Failing input**: `sources=['0'], directives={'0': 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 21, in <module>
    test_extension_cython_params_preserved_with_pyrex_kwargs()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 6, in test_extension_cython_params_preserved_with_pyrex_kwargs
    st.lists(st.text(min_size=1), min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/15/hypo.py", line 17, in test_extension_cython_params_preserved_with_pyrex_kwargs
    assert ext.cython_directives == directives
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_extension_cython_params_preserved_with_pyrex_kwargs(
    sources=['0'],  # or any other generated value
    directives={'0': 0},
)
```
</details>

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

print("cython_include_dirs:", ext.cython_include_dirs)
print("cython_directives:", ext.cython_directives)
print("cython_cplus:", ext.cython_cplus)
```

<details>

<summary>
Expected: cython_include_dirs and cython_directives preserved, Actual: reset to empty
</summary>
```
cython_include_dirs: []
cython_directives: {}
cython_cplus: True
```
</details>

## Why This Is A Bug

The Extension class is designed to provide backward compatibility by accepting both `pyrex_*` and `cython_*` parameters. The constructor correctly translates `pyrex_*` kwargs to `cython_*` kwargs (lines 42-45), but when recursively calling itself (line 47-63), it fails to pass through any explicitly provided `cython_*` constructor parameters. The early return on line 64 prevents the cython attribute initialization (lines 83-92) from ever executing.

This violates the expected behavior where users should be able to mix legacy `pyrex_*` kwargs with modern `cython_*` parameters. The documentation doesn't indicate that using any `pyrex_*` parameter would cause all `cython_*` parameters to be silently ignored. This creates a confusing situation where valid parameters are accepted but have no effect, with no error or warning provided to the user.

## Relevant Context

The bug exists in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/extension.py`. The issue stems from the backward compatibility code path that handles `pyrex_*` parameters (introduced for legacy support). When the code detects any `pyrex_*` kwargs, it performs a recursive call to the constructor but only passes through the standard distutils Extension parameters, omitting all the Cython-specific ones.

The `cython_cplus` parameter works correctly because it's been renamed from `pyrex_cplus` to `cython_cplus` in the kwargs dictionary before the recursive call, so it gets passed through in the `**kw` parameter. However, parameters like `cython_include_dirs` and `cython_directives` that were passed as explicit named parameters are lost.

## Proposed Fix

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