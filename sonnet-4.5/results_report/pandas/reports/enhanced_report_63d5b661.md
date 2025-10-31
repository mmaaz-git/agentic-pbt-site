# Bug Report: pandas.compat._optional.import_optional_dependency Version Checking Bypassed for Submodules

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Version checking is completely bypassed for submodules like `lxml.etree` because the code looks up version requirements using the parent module name ('lxml') in the VERSIONS dict, but VERSIONS contains the full submodule name ('lxml.etree') as the key.

## Property-Based Test

```python
"""
Property-based test that demonstrates the version checking bug
in pandas.compat._optional for submodules
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pandas.compat._optional import VERSIONS

@settings(max_examples=50)
@given(st.sampled_from(list(VERSIONS.keys())))
def test_version_dict_entries_use_correct_lookup_key(module_name):
    """
    Property: For every module in VERSIONS, the lookup key used in
    import_optional_dependency should find the version requirement.
    """
    if "." in module_name:
        parent = module_name.split(".")[0]
        in_versions_as_full_name = module_name in VERSIONS
        in_versions_as_parent = parent in VERSIONS

        if in_versions_as_full_name and not in_versions_as_parent:
            raise AssertionError(
                f"Bug: VERSIONS contains '{module_name}' but code will "
                f"look up '{parent}', causing version check to be skipped"
            )

if __name__ == "__main__":
    # Run the test
    test_version_dict_entries_use_correct_lookup_key()
```

<details>

<summary>
**Failing input**: `module_name='lxml.etree'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 32, in <module>
    test_version_dict_entries_use_correct_lookup_key()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 13, in test_version_dict_entries_use_correct_lookup_key
    @given(st.sampled_from(list(VERSIONS.keys())))
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 25, in test_version_dict_entries_use_correct_lookup_key
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: Bug: VERSIONS contains 'lxml.etree' but code will look up 'lxml', causing version check to be skipped
Falsifying example: test_version_dict_entries_use_correct_lookup_key(
    module_name='lxml.etree',
)
```
</details>

## Reproducing the Bug

```python
"""
Demonstrates the bug where version checking is skipped for
submodules like 'lxml.etree' in pandas.compat._optional
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat._optional import VERSIONS

# The bug: VERSIONS contains 'lxml.etree' but the code looks up 'lxml'
name = "lxml.etree"
parent = name.split(".")[0]  # This gives us 'lxml'

print("=== Bug Demonstration ===")
print(f"Module name: '{name}'")
print(f"Parent module (what code looks up): '{parent}'")
print()

# Show that 'lxml.etree' is in VERSIONS with a specific version requirement
print(f"Is '{name}' in VERSIONS? {name in VERSIONS}")
if name in VERSIONS:
    print(f"  Version requirement for '{name}': {VERSIONS[name]}")
print()

# Show that 'lxml' (the parent) is NOT in VERSIONS
print(f"Is '{parent}' in VERSIONS? {parent in VERSIONS}")
print(f"  VERSIONS.get('{parent}'): {VERSIONS.get(parent)}")
print()

# This is the actual bug - the code at line 148 does:
# minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
print("=== The Bug ===")
print(f"Code at line 148 uses: VERSIONS.get('{parent}') = {VERSIONS.get(parent)}")
print(f"It SHOULD use: VERSIONS.get('{name}') = {VERSIONS.get(name)}")
print()
print("Result: Version checking is completely skipped because minimum_version = None")
print("Expected: Version checking should validate lxml.etree >= 4.9.2")
```

<details>

<summary>
Version checking skipped for lxml.etree submodule
</summary>
```
=== Bug Demonstration ===
Module name: 'lxml.etree'
Parent module (what code looks up): 'lxml'

Is 'lxml.etree' in VERSIONS? True
  Version requirement for 'lxml.etree': 4.9.2

Is 'lxml' in VERSIONS? False
  VERSIONS.get('lxml'): None

=== The Bug ===
Code at line 148 uses: VERSIONS.get('lxml') = None
It SHOULD use: VERSIONS.get('lxml.etree') = 4.9.2

Result: Version checking is completely skipped because minimum_version = None
Expected: Version checking should validate lxml.etree >= 4.9.2
```
</details>

## Why This Is A Bug

The VERSIONS dict at line 30 explicitly specifies `"lxml.etree": "4.9.2"` to enforce a minimum version requirement for the lxml.etree submodule. The docstring for `import_optional_dependency` (lines 92-96) states: "By default, if a dependency is missing an ImportError with a nice message will be raised. If a dependency is present, but too old, we raise."

However, the code at line 148 in `_optional.py` incorrectly retrieves the minimum version:

```python
minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
```

For `name="lxml.etree"`, this code:
1. Splits the name to get `parent="lxml"` (line 142)
2. Looks up `VERSIONS.get("lxml")` which returns `None` (line 148)
3. Since `minimum_version` is `None`, the entire version validation block (lines 149-166) is skipped

This violates the documented behavior and the clear intent of having `"lxml.etree": "4.9.2"` in the VERSIONS dict. The bug means pandas will silently accept any version of lxml, even versions older than 4.9.2, which could lead to runtime failures or unexpected behavior when using XML/HTML parsing functionality.

## Relevant Context

The bug affects the version checking mechanism that pandas uses to ensure compatible versions of optional dependencies are installed. The VERSIONS dictionary (lines 17-56 in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/compat/_optional.py`) is carefully maintained to specify minimum version requirements for all optional dependencies.

The INSTALL_MAPPING dictionary (lines 61-71) correctly maps `"lxml.etree": "lxml"` for installation purposes, showing that pandas developers are aware of the submodule relationship and intend to handle it properly.

Currently, only `lxml.etree` is affected as it's the only entry in VERSIONS that is a submodule where the parent module name is not also present in the dictionary. This could affect any pandas functionality that relies on lxml for XML/HTML parsing, including `read_html()`, `read_xml()`, and `to_xml()` methods.

## Proposed Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -145,7 +145,7 @@ def import_optional_dependency(
         module_to_get = sys.modules[install_name]
     else:
         module_to_get = module
-    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
+    minimum_version = min_version if min_version is not None else VERSIONS.get(name)
     if minimum_version:
         version = get_version(module_to_get)
         if version and Version(version) < Version(minimum_version):
```