# Bug Report: FastAPI Dependencies Version Comparison Using String Instead of Semantic Versioning

**Target**: `fastapi.dependencies.utils.ensure_multipart_is_installed`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ensure_multipart_is_installed` function uses lexicographic string comparison (`__version__ > "0.0.12"`) instead of semantic version comparison, causing incorrect version validation for certain version numbers.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume, example
from packaging import version as pkg_version


def version_string_strategy():
    return st.builds(
        lambda major, minor, patch: f"{major}.{minor}.{patch}",
        major=st.integers(min_value=0, max_value=10),
        minor=st.integers(min_value=0, max_value=99),
        patch=st.integers(min_value=0, max_value=999)
    )


@given(v1=version_string_strategy(), v2=version_string_strategy())
@example(v1="0.0.9", v2="0.0.12")
@example(v1="0.0.100", v2="0.0.12")
def test_version_comparison_string_vs_semantic(v1, v2):
    """String comparison should match semantic versioning."""
    assume(v1 != v2)

    string_comparison = v1 > v2
    semantic_comparison = pkg_version.parse(v1) > pkg_version.parse(v2)

    assert string_comparison == semantic_comparison, \
        f"Mismatch: '{v1}' > '{v2}' = {string_comparison} (string), " \
        f"but semantically = {semantic_comparison}"
```

**Failing input**: `v1="0.0.9", v2="0.0.12"`

## Reproducing the Bug

```python
current_version = "0.0.9"
required_version = "0.0.12"

string_result = current_version > required_version
print(f"String comparison: '{current_version}' > '{required_version}' = {string_result}")

from packaging import version
semantic_result = version.parse(current_version) > version.parse(required_version)
print(f"Semantic comparison: {current_version} > {required_version} = {semantic_result}")

print(f"\nBug: {string_result} != {semantic_result}")
```

**Output:**
```
String comparison: '0.0.9' > '0.0.12' = True
Semantic comparison: 0.0.9 > 0.0.12 = False

Bug: True != False
```

## Why This Is A Bug

Line 96 in `fastapi/dependencies/utils.py` uses string comparison for version checking:

```python
assert __version__ > "0.0.12"
```

String comparison in Python is lexicographic (character-by-character), not semantic:
- `"0.0.9" > "0.0.12"` returns `True` (incorrect) because `"9" > "1"` in ASCII
- `"0.0.100" > "0.0.12"` returns `False` (incorrect) because `"1" < "2"` in ASCII

This causes:
1. **False acceptance**: Version 0.0.9 would pass the check despite being older than 0.0.12
2. **False rejection**: Version 0.0.100 would fail the check despite being newer than 0.0.12

## Fix

Use semantic version comparison with the `packaging` library:

```diff
def ensure_multipart_is_installed() -> None:
+    from packaging import version
+
     try:
         from python_multipart import __version__

-        assert __version__ > "0.0.12"
+        assert version.parse(__version__) > version.parse("0.0.12")
     except (ImportError, AssertionError):
         # ... rest of error handling
```

Alternative fix using `packaging.version.Version`:

```diff
def ensure_multipart_is_installed() -> None:
     try:
         from python_multipart import __version__
+        from packaging.version import Version

-        assert __version__ > "0.0.12"
+        assert Version(__version__) > Version("0.0.12")
     except (ImportError, AssertionError):
         # ... rest of error handling
```