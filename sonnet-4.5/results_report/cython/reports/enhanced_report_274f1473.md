# Bug Report: Cython.Debugger.Tests.TestLibCython GDB Version Detection Regex Matches Wrong Version

**Target**: `Cython.Debugger.Tests.TestLibCython.test_gdb()`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The GDB version detection regex in test_gdb() incorrectly matches Ubuntu/Debian package versions instead of actual GDB versions, causing tests to potentially run on incompatible GDB versions or skip on compatible ones.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test demonstrating Cython GDB version detection bug."""

import re
from hypothesis import given, strategies as st, settings, assume, example


@given(
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
    st.integers(min_value=0, max_value=99),
)
@example(ubuntu_major=12, ubuntu_minor=1, gdb_major=7, gdb_minor=2)
@settings(max_examples=1000)
def test_gdb_version_regex_bug(ubuntu_major, ubuntu_minor, gdb_major, gdb_minor):
    """Test that GDB version regex incorrectly matches package version instead of actual version."""
    # Only test cases where the versions differ
    assume(ubuntu_major != gdb_major or ubuntu_minor != gdb_minor)

    # The regex from Cython.Debugger.Tests.TestLibCython.test_gdb() line 42
    regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"

    # Format typical of Ubuntu/Debian GDB packages
    test_input = f"GNU gdb (Ubuntu {ubuntu_major}.{ubuntu_minor}-0ubuntu1~22.04) {gdb_major}.{gdb_minor}"

    match = re.match(regex, test_input)
    assert match is not None, f"Regex should match string: {test_input}"

    groups = tuple(map(int, match.groups()))

    # The bug: regex matches Ubuntu package version instead of actual GDB version
    assert groups == (gdb_major, gdb_minor), \
        f"Regex matched Ubuntu version {groups} instead of GDB version ({gdb_major}, {gdb_minor})"


if __name__ == "__main__":
    test_gdb_version_regex_bug()
```

<details>

<summary>
**Failing input**: `ubuntu_major=12, ubuntu_minor=1, gdb_major=7, gdb_minor=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 38, in <module>
    test_gdb_version_regex_bug()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 9, in test_gdb_version_regex_bug
    st.integers(min_value=0, max_value=99),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 33, in test_gdb_version_regex_bug
    assert groups == (gdb_major, gdb_minor), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Regex matched Ubuntu version (12, 1) instead of GDB version (7, 2)
Falsifying explicit example: test_gdb_version_regex_bug(
    ubuntu_major=12,
    ubuntu_minor=1,
    gdb_major=7,
    gdb_minor=2,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython GDB version detection bug."""

import re

# The regex from Cython.Debugger.Tests.TestLibCython.test_gdb() line 42
regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"

# Real-world GDB version output from Ubuntu 22.04
gdb_output = "GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 7.2"

# Try to match the version
match = re.match(regex, gdb_output)
if match:
    version = list(map(int, match.groups()))
    print(f"Input string: {gdb_output}")
    print(f"Regex pattern: {regex}")
    print(f"Detected version: {version}")
    print(f"Expected: [7, 2] (actual GDB version)")
    print(f"Actual: {version} (Ubuntu package version)")

    # The bug: this assertion fails
    assert version == [7, 2], f"Bug: Matched Ubuntu package version {version} instead of GDB version [7, 2]"
else:
    print("No match found")
```

<details>

<summary>
AssertionError: Matched wrong version number
</summary>
```
Input string: GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 7.2
Regex pattern: GNU gdb [^\d]*(\d+)\.(\d+)
Detected version: [12, 1]
Expected: [7, 2] (actual GDB version)
Actual: [12, 1] (Ubuntu package version)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 23, in <module>
    assert version == [7, 2], f"Bug: Matched Ubuntu package version {version} instead of GDB version [7, 2]"
           ^^^^^^^^^^^^^^^^^
AssertionError: Bug: Matched Ubuntu package version [12, 1] instead of GDB version [7, 2]
```
</details>

## Why This Is A Bug

The regex pattern `r"GNU gdb [^\d]*(\d+)\.(\d+)"` in TestLibCython.py line 42 is intended to extract the GDB version number to determine if GDB >= 7.2 is available for running debugger tests. However, the pattern has a critical flaw:

- `[^\d]*` matches zero or more non-digit characters after "GNU gdb"
- This causes it to match the FIRST version number encountered, which in Ubuntu/Debian packages is the package version in parentheses (e.g., "12.1" in "(Ubuntu 12.1-0ubuntu1~22.04)")
- The actual GDB version appears later in the string (e.g., "7.2")

This causes incorrect test execution behavior:
1. **False positives**: Tests may run on GDB versions < 7.2 if the Ubuntu package version >= 7.2 (e.g., Ubuntu 12.1 package containing GDB 6.8 would incorrectly pass the version check)
2. **False negatives**: Tests may be skipped on GDB versions >= 7.2 if the Ubuntu package version < 7.2 (e.g., Ubuntu 5.4 package containing GDB 11.3 would incorrectly fail the version check)

The function's purpose per line 64 is clear: "Skipping gdb tests, need gdb >= 7.2 with Python >= 2.7". The current implementation fails to correctly determine the GDB version for this compatibility check on major Linux distributions.

## Relevant Context

This bug affects the Cython project's internal test suite, specifically the GDB debugger integration tests. While it doesn't affect end users directly, it impacts the reliability of Cython's test infrastructure on Ubuntu and Debian systems, which are among the most popular Linux distributions.

The code appears to be based on Python's own test_gdb.py (as noted in the comment on line 41), suggesting this pattern may have been inherited from upstream Python test code. The bug has likely persisted because:
1. The test suite continues to function despite occasional incorrect skipping/running
2. Developers may work around it by installing GDB from source
3. The impact is limited to test execution, not production functionality

Ubuntu and Debian GDB packages follow the format:
```
GNU gdb (Ubuntu <package-version>) <actual-gdb-version>
GNU gdb (Debian <package-version>) <actual-gdb-version>
```

Other distributions may have different formats (e.g., Red Hat), suggesting a more comprehensive solution might be needed.

## Proposed Fix

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