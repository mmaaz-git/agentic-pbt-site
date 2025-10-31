# Bug Report: Cython.Distutils.Extension Pyrex Options Override Explicit Cython Parameters

**Target**: `Cython.Distutils.extension.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When pyrex_* backward compatibility options are provided alongside explicit cython_* parameters, the Extension.__init__ method silently discards the explicit cython_* parameters due to a flawed recursive call pattern.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension


@given(
    pyrex_val=st.booleans(),
    cython_val=st.booleans()
)
def test_explicit_cython_params_not_overridden_by_pyrex(pyrex_val, cython_val):
    ext = Extension(
        "test",
        ["test.pyx"],
        **{f"pyrex_gdb": pyrex_val},
        cython_gdb=cython_val
    )

    assert ext.cython_gdb == cython_val, \
        f"Explicit cython_gdb={cython_val} was overridden by pyrex_gdb={pyrex_val}"


@given(
    pyrex_list=st.lists(st.text(min_size=1, max_size=20), max_size=3),
    cython_list=st.lists(st.text(min_size=1, max_size=20), max_size=3)
)
def test_explicit_cython_include_dirs_not_overridden_by_pyrex(pyrex_list, cython_list):
    ext = Extension(
        "test",
        ["test.pyx"],
        **{"pyrex_include_dirs": pyrex_list},
        cython_include_dirs=cython_list
    )

    assert ext.cython_include_dirs == cython_list
```

**Failing input**: `pyrex_val=True, cython_val=False`

## Reproducing the Bug

```python
from Cython.Distutils import Extension

ext = Extension(
    "test",
    ["test.pyx"],
    pyrex_gdb=True,
    cython_gdb=False
)

print(f"Expected: cython_gdb=False (explicit parameter)")
print(f"Actual: cython_gdb={ext.cython_gdb}")
assert ext.cython_gdb == False
```

## Why This Is A Bug

The pyrex_* to cython_* translation logic (lines 42-64) has a critical flaw:

1. It converts `pyrex_x` to `cython_x` in the **kw dict (line 45)
2. When any pyrex_* option is present, it recursively calls Extension.__init__ (lines 47-63)
3. This recursive call passes **kw (containing converted pyrex options) but **omits all explicit cython_* parameters**
4. Result: Explicit `cython_gdb=False` is lost, replaced by `cython_gdb=True` from converted `pyrex_gdb`

Explicit parameters should always take precedence over backward-compatibility translations. Users who explicitly specify cython_* parameters expect them to be honored, regardless of whether deprecated pyrex_* options are also present (perhaps from legacy configuration files).

## Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -42,23 +42,32 @@ class Extension(_Extension.Extension):
         for key in list(kw):
             if key.startswith('pyrex_'):
                 had_pyrex_options = True
-                kw['cython' + key[5:]] = kw.pop(key)
+                cython_key = 'cython' + key[5:]
+                # Only use pyrex value if cython equivalent wasn't explicitly provided
+                if cython_key not in kw:
+                    kw[cython_key] = kw.pop(key)
+                else:
+                    kw.pop(key)  # Discard pyrex option, keep explicit cython option
         if had_pyrex_options:
+            # Merge explicit cython_* parameters into kw, giving them precedence
+            for param in ['cython_include_dirs', 'cython_directives', 'cython_create_listing',
+                          'cython_line_directives', 'cython_cplus', 'cython_c_in_temp',
+                          'cython_gen_pxi', 'cython_gdb', 'cython_compile_time_env']:
+                if param in locals() and locals()[param] is not None:
+                    kw[param] = locals()[param]
             Extension.__init__(
                 self, name, sources,
                 include_dirs=include_dirs,
                 define_macros=define_macros,
```

Note: A cleaner fix would be to avoid the recursive call pattern entirely and just update the parameters before the main initialization.