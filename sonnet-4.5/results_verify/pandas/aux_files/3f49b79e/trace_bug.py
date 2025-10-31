#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

# Let's trace through what happens when import_optional_dependency is called with "lxml.etree"

print("=== Tracing import_optional_dependency('lxml.etree') ===\n")

from pandas.compat._optional import VERSIONS, import_optional_dependency

name = "lxml.etree"
print(f"1. Function called with name = '{name}'")

# Line 142: parent = name.split(".")[0]
parent = name.split(".")[0]
print(f"2. Line 142: parent = name.split('.')[0] = '{parent}'")

# Line 148: minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
# Since min_version is typically None when not explicitly passed
min_version = None
minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
print(f"3. Line 148: minimum_version = VERSIONS.get('{parent}') = {minimum_version}")

print(f"\nThe problem:")
print(f"  - VERSIONS has key 'lxml.etree' with value '4.9.2'")
print(f"  - But code looks up 'lxml' which doesn't exist in VERSIONS")
print(f"  - So minimum_version becomes None and version check is skipped!")

print(f"\nWhat the code SHOULD do:")
correct_minimum_version = VERSIONS.get(name)
print(f"  - Look up VERSIONS.get('{name}') = {correct_minimum_version}")

# Let's also check if there are other modules with this issue
print("\n=== Checking for other affected modules ===")
for module_name in VERSIONS.keys():
    if "." in module_name:
        parent_mod = module_name.split(".")[0]
        if parent_mod not in VERSIONS:
            print(f"  - {module_name}: will incorrectly look up '{parent_mod}' instead")