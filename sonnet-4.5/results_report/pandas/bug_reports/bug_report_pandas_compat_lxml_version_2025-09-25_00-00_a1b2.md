# Bug Report: pandas.compat._optional Version Checking Skipped for Submodules

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Version checking is completely bypassed for submodules like `lxml.etree` because the code looks up version requirements using the parent module name in the VERSIONS dict, but VERSIONS contains the full submodule name as the key.

## Property-Based Test

```python
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
```

**Failing input**: `module_name='lxml.etree'`

## Reproducing the Bug

```python
from pandas.compat._optional import VERSIONS

name = "lxml.etree"
parent = name.split(".")[0]

assert "lxml.etree" in VERSIONS
assert VERSIONS["lxml.etree"] == "4.9.2"
assert "lxml" not in VERSIONS
assert VERSIONS.get(parent) is None

print("Version checking is skipped because:")
print(f"  Code uses: VERSIONS.get('{parent}') = {VERSIONS.get(parent)}")
print(f"  Should use: VERSIONS.get('{name}') = {VERSIONS.get(name)}")
```

## Why This Is A Bug

The VERSIONS dict explicitly specifies `"lxml.etree": "4.9.2"` to enforce a minimum version requirement. However, the code at line 148 in `_optional.py` does:

```python
minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
```

For `name="lxml.etree"`, this computes `parent="lxml"` and looks up `VERSIONS.get("lxml")`, which returns `None` because the actual key is `"lxml.etree"`. This causes the version checking block to be skipped entirely, defeating the purpose of having the entry in VERSIONS.

This affects any submodule in VERSIONS where the parent module is not also present. Currently, this includes `lxml.etree`.

## Fix

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