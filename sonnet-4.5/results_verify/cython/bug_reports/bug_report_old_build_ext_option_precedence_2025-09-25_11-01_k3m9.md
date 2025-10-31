# Bug Report: Cython.Distutils.old_build_ext Option Precedence Violation

**Target**: `Cython.Distutils.old_build_ext.cython_sources`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cython_sources` method uses `or` operators for configuration options, which means falsy command-level values cannot override truthy extension values, violating the expected precedence.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Distutils import Extension
from Cython.Distutils.old_build_ext import old_build_ext
from distutils.dist import Distribution


@given(st.sampled_from(['cython_cplus', 'cython_gen_pxi']))
def test_command_level_zero_ignored(option_name):
    dist = Distribution()
    builder = old_build_ext(dist)
    builder.initialize_options()
    builder.finalize_options()

    setattr(builder, option_name, 1)
    ext = Extension('test', ['test.pyx'])
    setattr(ext, option_name, 0)

    option_value = getattr(builder, option_name) or getattr(ext, option_name, 0)

    assert option_value == 1
```

**Failing input**: N/A - test demonstrates expected behavior is impossible with current code

## Reproducing the Bug

```python
from Cython.Distutils import Extension
from Cython.Distutils.old_build_ext import old_build_ext
from distutils.dist import Distribution

dist = Distribution()
builder = old_build_ext(dist)
builder.initialize_options()

builder.cython_gen_pxi = 0
ext = Extension('test', ['test.pyx'], cython_gen_pxi=1)

computed = builder.cython_gen_pxi or getattr(ext, 'cython_gen_pxi', 0)

print(f"Command-level (builder): {builder.cython_gen_pxi}")
print(f"Extension-level: {ext.cython_gen_pxi}")
print(f"Computed value: {computed}")
print(f"Expected: 0 (command-level should win)")
print(f"Actual: {computed} (extension-level wins!)")
```

Output:
```
Command-level (builder): 0
Extension-level: 1
Computed value: 1
Expected: 0 (command-level should win)
Actual: 1 (extension-level wins!)
```

## Why This Is A Bug

While boolean options typically follow "any True wins" semantics, the code's structure suggests command-level options should take precedence. The pattern `self.option or extension.option` means extension values override command-level zeros/falses, which is backwards.

However, this is marked as **Low severity** because:
1. For boolean flags, it's unclear whether "0 at command level, 1 at extension level" is a realistic use case
2. The module is deprecated (line 6: "Note that this module is deprecated")
3. The modern `build_ext` has the same issue in `get_extension_attr`, suggesting this might be intentional legacy behavior

Affected options (old_build_ext.py:223-234):
- `cython_create_listing`, `cython_line_directives`, `no_c_in_traceback`
- `cython_cplus`, `cython_gen_pxi`, `cython_gdb`, `cython_compile_time_env`

## Fix

Defer to the fix for `build_ext.get_extension_attr` since both issues are related. The modern approach would use `hasattr(self, option_name)` checks or implement a proper option merging strategy that respects explicit zeros.