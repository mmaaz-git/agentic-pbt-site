# Bug Report: Cython.Distutils.Extension Pyrex Parameters Override Explicit Cython Parameters

**Target**: `Cython.Distutils.extension.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When both deprecated pyrex_* backward compatibility options and explicit cython_* parameters are provided to Extension.__init__, the explicit cython_* parameters are silently discarded due to a flawed recursive call implementation.

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


if __name__ == "__main__":
    # Run the tests
    test_explicit_cython_params_not_overridden_by_pyrex()
    test_explicit_cython_include_dirs_not_overridden_by_pyrex()
```

<details>

<summary>
**Failing input**: `pyrex_val=False, cython_val=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 38, in <module>
    test_explicit_cython_params_not_overridden_by_pyrex()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 6, in test_explicit_cython_params_not_overridden_by_pyrex
    pyrex_val=st.booleans(),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 17, in test_explicit_cython_params_not_overridden_by_pyrex
    assert ext.cython_gdb == cython_val, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Explicit cython_gdb=True was overridden by pyrex_gdb=False
Falsifying example: test_explicit_cython_params_not_overridden_by_pyrex(
    pyrex_val=False,
    cython_val=True,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/12/hypo.py:18
```
</details>

## Reproducing the Bug

```python
from Cython.Distutils import Extension

# Test case: Both pyrex_gdb and cython_gdb provided
# Expected: cython_gdb=False (explicit parameter should take precedence)
# Actual: cython_gdb will be True (pyrex_gdb value overrides)

ext = Extension(
    "test",
    ["test.pyx"],
    pyrex_gdb=True,
    cython_gdb=False
)

print(f"Expected: cython_gdb=False (explicit parameter)")
print(f"Actual: cython_gdb={ext.cython_gdb}")

# This assertion will fail, demonstrating the bug
try:
    assert ext.cython_gdb == False, f"Bug: explicit cython_gdb=False was overridden by pyrex_gdb=True"
    print("✓ Test passed: Explicit cython_gdb parameter was respected")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
```

<details>

<summary>
AssertionError: explicit cython_gdb parameter was overridden
</summary>
```
Expected: cython_gdb=False (explicit parameter)
Actual: cython_gdb=True
✗ Test failed: Bug: explicit cython_gdb=False was overridden by pyrex_gdb=True
```
</details>

## Why This Is A Bug

This violates the fundamental principle that explicit parameters should take precedence over implicit translations. The bug occurs because of a flawed implementation in the Extension.__init__ method (lines 42-64 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/extension.py`):

1. When pyrex_* options are detected, the code converts them to cython_* in the **kw dictionary (line 45)
2. It then makes a recursive call to Extension.__init__ (lines 47-63) passing the converted **kw
3. **Critical Issue**: The recursive call passes only the positional parameters and **kw, but completely omits all the explicit cython_* parameters that were passed as named arguments
4. The early return on line 64 prevents lines 83-92 from executing, which would have set the cython attributes
5. Result: Any explicit cython_* parameters are silently discarded when any pyrex_* option is present

This behavior is undocumented and counterintuitive. Users migrating from Pyrex to Cython may have configuration files with pyrex_* options while explicitly setting cython_* options in code. They would reasonably expect their explicit parameters to be honored, not silently overridden by backward-compatibility translations.

## Relevant Context

- **Migration Scenario**: This bug primarily affects users migrating from Pyrex to Cython who may have both old configuration (pyrex_*) and new code (cython_*) parameters
- **Silent Failure**: The bug is particularly problematic because it silently ignores user input without any warning or error
- **Scope**: Affects all cython_* parameters including cython_include_dirs, cython_directives, cython_create_listing, cython_line_directives, cython_cplus, cython_c_in_temp, cython_gen_pxi, cython_gdb, and cython_compile_time_env
- **Documentation**: The source code only contains a brief comment "Translate pyrex_X to cython_X for backwards compatibility" with no mention of precedence rules
- **Workaround**: Users can avoid this bug by using only cython_* parameters and removing all pyrex_* options

## Proposed Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -42,23 +42,33 @@ class Extension(_Extension.Extension):
         for key in list(kw):
             if key.startswith('pyrex_'):
                 had_pyrex_options = True
-                kw['cython' + key[5:]] = kw.pop(key)
+                cython_key = 'cython' + key[5:]
+                # Only use pyrex value if cython equivalent wasn't explicitly provided
+                if cython_key not in kw:
+                    kw[cython_key] = kw.pop(key)
+                else:
+                    kw.pop(key)  # Remove pyrex option, keep explicit cython option
         if had_pyrex_options:
+            # Pass explicit cython_* parameters to the recursive call
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
+                cython_compile_time_env=cython_compile_time_env,
                 no_c_in_traceback=no_c_in_traceback,
                 **kw)
             return
```