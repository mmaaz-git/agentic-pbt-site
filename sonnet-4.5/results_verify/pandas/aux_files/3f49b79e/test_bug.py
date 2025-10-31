#!/usr/bin/env python3

# First, run the hypothesis test
print("=== Running Hypothesis Test ===")
from hypothesis import given, strategies as st, settings
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
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

try:
    test_version_dict_entries_use_correct_lookup_key()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Unexpected error in hypothesis test: {e}")

print("\n=== Running Reproduction Code ===")

# Now run the reproduction code
from pandas.compat._optional import VERSIONS

name = "lxml.etree"
parent = name.split(".")[0]

print(f"Testing module: {name}")
print(f"Parent module: {parent}")
print()

try:
    assert "lxml.etree" in VERSIONS
    print(f"✓ 'lxml.etree' is in VERSIONS")

    assert VERSIONS["lxml.etree"] == "4.9.2"
    print(f"✓ VERSIONS['lxml.etree'] = '4.9.2'")

    assert "lxml" not in VERSIONS
    print(f"✓ 'lxml' is NOT in VERSIONS")

    assert VERSIONS.get(parent) is None
    print(f"✓ VERSIONS.get('{parent}') returns None")

    print("\nVersion checking is skipped because:")
    print(f"  Code uses: VERSIONS.get('{parent}') = {VERSIONS.get(parent)}")
    print(f"  Should use: VERSIONS.get('{name}') = {VERSIONS.get(name)}")

    print("\n✓ All assertions passed - bug confirmed!")

except AssertionError as e:
    print(f"✗ Assertion failed: {e}")