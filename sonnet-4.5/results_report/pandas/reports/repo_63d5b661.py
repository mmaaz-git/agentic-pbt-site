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