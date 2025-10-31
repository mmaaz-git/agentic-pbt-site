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