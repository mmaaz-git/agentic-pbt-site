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

if __name__ == "__main__":
    # Run the test
    test_gdb_version_regex_bug()