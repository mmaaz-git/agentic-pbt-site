# Bug Report: Cython.Distutils.build_ext.get_extension_attr Ignores Falsy Command-Line Values

**Target**: `Cython.Distutils.build_ext.get_extension_attr`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_extension_attr` method incorrectly uses the `or` operator to choose between command-line and extension settings, causing falsy values (0, False, "", [], {}) from command-line options to be ignored in favor of extension-level settings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


@given(
    st.one_of(st.just(0), st.just(False), st.just(""), st.just([]))
)
@settings(max_examples=100)
def test_get_extension_attr_falsy_bug(falsy_value):
    dist = Distribution()
    build_ext_instance = build_ext(dist)
    build_ext_instance.initialize_options()
    build_ext_instance.finalize_options()

    class MockExtension:
        pass

    ext = MockExtension()

    setattr(build_ext_instance, 'test_option', falsy_value)
    setattr(ext, 'test_option', "extension_value")

    result = build_ext_instance.get_extension_attr(ext, 'test_option', default="default")

    assert result == falsy_value, f"Expected {repr(falsy_value)}, got {repr(result)}"


if __name__ == "__main__":
    test_get_extension_attr_falsy_bug()
```

<details>

<summary>
**Failing input**: `falsy_value=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 30, in <module>
    test_get_extension_attr_falsy_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 7, in test_get_extension_attr_falsy_bug
    st.one_of(st.just(0), st.just(False), st.just(""), st.just([]))
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 26, in test_get_extension_attr_falsy_bug
    assert result == falsy_value, f"Expected {repr(falsy_value)}, got {repr(result)}"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 0, got 'extension_value'
Falsifying example: test_get_extension_attr_falsy_bug(
    falsy_value=0,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Distutils.build_ext import build_ext
from distutils.dist import Distribution


class MockExtension:
    cython_cplus = 1


dist = Distribution()
build_ext_instance = build_ext(dist)
build_ext_instance.initialize_options()
build_ext_instance.finalize_options()

# Set command-line option to 0 (disable C++)
build_ext_instance.cython_cplus = 0

ext = MockExtension()

# Should return 0 (command-line value), but returns 1 (extension value)
result = build_ext_instance.get_extension_attr(ext, 'cython_cplus')

print(f"Command-line setting (build_ext_instance.cython_cplus): {build_ext_instance.cython_cplus}")
print(f"Extension setting (ext.cython_cplus): {ext.cython_cplus}")
print(f"Result from get_extension_attr: {result}")
print(f"Expected: 0 (command-line should override)")
print(f"Actual: {result}")
print(f"Bug confirmed: {result != 0}")
```

<details>

<summary>
Command-line option 0 incorrectly overridden by extension value 1
</summary>
```
Command-line setting (build_ext_instance.cython_cplus): 0
Extension setting (ext.cython_cplus): 1
Result from get_extension_attr: 1
Expected: 0 (command-line should override)
Actual: 1
Bug confirmed: True
```
</details>

## Why This Is A Bug

The `get_extension_attr` method at line 81 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Distutils/build_ext.py` uses:

```python
return getattr(self, option_name) or getattr(extension, option_name, default)
```

This violates standard build system conventions where command-line options should override configuration/extension settings. The `or` operator treats any falsy value (0, False, "", [], {}) as "not set", causing the method to incorrectly fall through to the extension's value.

Real-world impacts include:
- Cannot disable C++ compilation by setting `cython_cplus=0` when an extension has `cython_cplus=1`
- Cannot clear compile-time environment variables with `cython_compile_time_env={}` when extension has values
- Cannot override any boolean flag set to False/0 at the command-line level

This behavior contradicts the expected precedence order found in other Python build tools where more specific settings (command-line) override less specific ones (extension/configuration).

## Relevant Context

The bug is located in the `build_ext` class which inherits from distutils' build_ext. The class defines various Cython-specific options in `initialize_options()` (lines 57-68), with boolean options initialized to 0 and object options to None.

The `get_extension_attr` method is called multiple times in `build_extension()` (lines 111-128) to retrieve options that can come from either the command-line (stored on the build_ext instance) or from individual Extension objects. This method is critical for determining compilation settings.

The current implementation makes it impossible to explicitly set a falsy value at the command-line level that differs from an extension's setting, breaking the fundamental principle of command-line override precedence.

## Proposed Fix

```diff
--- a/Cython/Distutils/build_ext.py
+++ b/Cython/Distutils/build_ext.py
@@ -78,7 +78,8 @@ class build_ext(_build_ext):
             self.cython_directives = {}

     def get_extension_attr(self, extension, option_name, default=False):
-        return getattr(self, option_name) or getattr(extension, option_name, default)
+        self_value = getattr(self, option_name, None)
+        return self_value if self_value is not None else getattr(extension, option_name, default)
```