# Bug Report: Cython.Distutils.Extension Silently Discards cython_* Parameters When pyrex_* Parameters Are Present

**Target**: `Cython.Distutils.extension.Extension.__init__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When any `pyrex_*` parameter is passed to Extension's constructor alongside explicit `cython_*` parameters, the explicit `cython_*` parameters are silently discarded and reset to their default values, causing unexpected behavior and data loss.

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

# Run the test
if __name__ == "__main__":
    test_cython_params_preserved_with_pyrex()
```

<details>

<summary>
**Failing input**: `name='A', cython_value=True, pyrex_param='pyrex_include_dirs'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 24, in <module>
    test_cython_params_preserved_with_pyrex()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 7, in test_cython_params_preserved_with_pyrex
    name=valid_identifiers,
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 19, in test_cython_params_preserved_with_pyrex
    assert ext.cython_gdb == cython_value, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: cython_gdb should be True, but got False when pyrex_include_dirs is also present
Falsifying example: test_cython_params_preserved_with_pyrex(
    name='A',
    cython_value=True,
    pyrex_param='pyrex_include_dirs',
)
```
</details>

## Reproducing the Bug

```python
from Cython.Distutils import Extension

# Create an Extension with both pyrex_* and cython_* parameters
ext = Extension(
    "mymodule",
    ["mymodule.pyx"],
    pyrex_include_dirs=["/legacy/path"],  # This triggers the pyrex compatibility code
    cython_gdb=True                        # This should be True but gets lost
)

print(f"Expected cython_gdb: True")
print(f"Actual cython_gdb: {ext.cython_gdb}")

# This assertion will fail, demonstrating the bug
try:
    assert ext.cython_gdb == True, "cython_gdb should be True"
    print("PASS: cython_gdb is correctly set to True")
except AssertionError as e:
    print(f"FAIL: {e}")
```

<details>

<summary>
Output demonstrating the parameter is lost
</summary>
```
Expected cython_gdb: True
Actual cython_gdb: False
FAIL: cython_gdb should be True
```
</details>

## Why This Is A Bug

The backwards compatibility code in `extension.py` (lines 40-64) is designed to translate `pyrex_*` parameters to their `cython_*` equivalents for users migrating from the obsolete Pyrex system. However, when any `pyrex_*` parameter is detected, the code makes a recursive call to `Extension.__init__()` that fails to pass along the explicitly-provided `cython_*` parameters.

Specifically, the issue occurs in this flow:

1. User provides both `pyrex_include_dirs=["/old"]` and `cython_gdb=True`
2. The code detects the pyrex option and converts it to `cython_include_dirs` in the `kw` dictionary (line 45)
3. It then makes a recursive call to `Extension.__init__()` (lines 47-63)
4. **Critical bug**: The recursive call only passes the standard distutils parameters and `no_c_in_traceback`, plus `**kw`
5. The `cython_gdb` parameter (and all other `cython_*` parameters) are NOT included in this recursive call because:
   - They are not in `kw` (they were passed as explicit named parameters to the original call)
   - They are not in the explicit parameter list passed to the recursive call
6. In the recursive call, `cython_gdb` gets its default value of `False`, silently discarding the user's explicit `True` value

This violates the principle of least surprise - users reasonably expect that explicitly-set parameters will be honored. The silent loss of configuration is particularly problematic for users attempting gradual migration from pyrex to cython who may mix old and new parameter names during the transition.

## Relevant Context

The bug affects ALL cython-specific parameters when ANY pyrex parameter is present:
- `cython_include_dirs`
- `cython_directives`
- `cython_create_listing`
- `cython_line_directives`
- `cython_cplus`
- `cython_c_in_temp`
- `cython_gen_pxi`
- `cython_gdb`
- `cython_compile_time_env`

The source code at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/extension.py` shows that while pyrex has been deprecated for many years, the backwards compatibility layer is still present and should function correctly when used.

Documentation: The Cython documentation does not explicitly specify the expected behavior when mixing pyrex and cython parameters, but the silent discarding of user-specified values is clearly unintended behavior.

## Proposed Fix

```diff
--- a/Cython/Distutils/extension.py
+++ b/Cython/Distutils/extension.py
@@ -44,6 +44,17 @@ class Extension(_Extension.Extension):
                 had_pyrex_options = True
                 kw['cython' + key[5:]] = kw.pop(key)
         if had_pyrex_options:
+            # Ensure explicit cython_* parameters are preserved in recursive call
+            if cython_include_dirs is not None:
+                kw['cython_include_dirs'] = cython_include_dirs
+            if cython_directives is not None:
+                kw['cython_directives'] = cython_directives
+            if cython_compile_time_env is not None:
+                kw['cython_compile_time_env'] = cython_compile_time_env
+            kw['cython_create_listing'] = cython_create_listing
+            kw['cython_line_directives'] = cython_line_directives
+            kw['cython_cplus'] = cython_cplus
+            kw['cython_c_in_temp'] = cython_c_in_temp
+            kw['cython_gen_pxi'] = cython_gen_pxi
+            kw['cython_gdb'] = cython_gdb
             Extension.__init__(
                 self, name, sources,
```