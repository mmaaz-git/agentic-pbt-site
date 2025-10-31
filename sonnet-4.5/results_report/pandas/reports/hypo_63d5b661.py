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