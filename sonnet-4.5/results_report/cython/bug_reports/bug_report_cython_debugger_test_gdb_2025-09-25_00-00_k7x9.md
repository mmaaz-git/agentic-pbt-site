# Bug Report: Cython.Debugger.Tests.TestLibCython test_gdb() Version Regex

**Target**: `Cython.Debugger.Tests.TestLibCython.test_gdb()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The GDB version detection regex incorrectly matches the Ubuntu/Debian package version instead of the actual GDB version, causing tests to run on incompatible GDB versions or skip on compatible ones.

## Property-Based Test

```python
import re
from hypothesis import given, strategies as st, settings, assume


@given(
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
)
@settings(max_examples=1000)
def test_gdb_version_regex_bug(ubuntu_major, ubuntu_minor, gdb_major, gdb_minor):
    assume(ubuntu_major != gdb_major or ubuntu_minor != gdb_minor)

    regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"
    test_input = f"GNU gdb (Ubuntu {ubuntu_major}.{ubuntu_minor}-0ubuntu1~22.04) {gdb_major}.{gdb_minor}"

    match = re.match(regex, test_input)
    assert match is not None
    groups = tuple(map(int, match.groups()))

    assert groups == (gdb_major, gdb_minor), \
        f"Regex matched Ubuntu version {groups} instead of GDB version ({gdb_major}, {gdb_minor})"
```

**Failing input**: `ubuntu_major=12, ubuntu_minor=1, gdb_major=7, gdb_minor=2`

## Reproducing the Bug

```python
import re

regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"
gdb_output = "GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 7.2"

match = re.match(regex, gdb_output)
version = list(map(int, match.groups()))

print(f"Detected version: {version}")
print(f"Expected: [7, 2]")
print(f"Actual: [12, 1]")
assert version == [7, 2], f"Bug: Matched Ubuntu version {version} instead of GDB version [7, 2]"
```

## Why This Is A Bug

The regex `r"GNU gdb [^\d]*(\d+)\.(\d+)"` matches the first version number after "GNU gdb", which is the package version in parentheses (e.g., "12.1" in "(Ubuntu 12.1...)") rather than the actual GDB version that appears later (e.g., "7.2").

This causes incorrect behavior:
- Tests may run on GDB < 7.2 if package version >= 7.2 (e.g., Ubuntu 12.1 package containing GDB 6.8)
- Tests may be skipped on GDB >= 7.2 if package version < 7.2 (e.g., Ubuntu 5.4 package containing GDB 11.3)

## Fix

```diff
--- a/Cython/Debugger/Tests/TestLibCython.py
+++ b/Cython/Debugger/Tests/TestLibCython.py
@@ -39,7 +39,7 @@ def test_gdb():
     else:
         stdout, _ = p.communicate()
         # Based on Lib/test/test_gdb.py
-        regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"
+        regex = r"GNU gdb.*[^\d](\d+)\.(\d+)"
         gdb_version = re.match(regex, stdout.decode('ascii', 'ignore'))

     if gdb_version:
```

The fix changes `[^\d]*` to `.*[^\d]`, using greedy matching to skip all content and backtrack to find the last version number on the first line.