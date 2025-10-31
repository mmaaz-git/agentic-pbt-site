#!/usr/bin/env python3
"""Test the reported bug with Cython DistutilsInfo comment handling"""

import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages")

# Test 1: Property-based test
print("=" * 60)
print("Test 1: Property-based test with hypothesis")
print("=" * 60)

from hypothesis import given, strategies as st, assume, settings
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1)))
@settings(max_examples=1000)
def test_parse_list_space_separated_count(items):
    assume(all(item.strip() for item in items))
    assume(all(' ' not in item and ',' not in item and '"' not in item and "'" not in item for item in items))

    list_str = ' '.join(items)
    result = parse_list(list_str)
    assert len(result) == len(items), f"Expected {len(items)} items, got {len(result)}. Input: {items}, Result: {result}"

# Test with the specific failing input
print("\nTesting specific failing input: items=['#', '0']")
items = ['#', '0']
list_str = ' '.join(items)
result = parse_list(list_str)
print(f"Input items: {items}")
print(f"Input string: '{list_str}'")
print(f"Result: {result}")
print(f"Result length: {len(result)}")
print(f"Expected length: {len(items)}")

# Test 2: DistutilsInfo with inline comments
print("\n" + "=" * 60)
print("Test 2: DistutilsInfo with inline comments")
print("=" * 60)

from Cython.Build.Dependencies import DistutilsInfo

source = """
# distutils: libraries = foo # this is a comment
"""

info = DistutilsInfo(source)
libraries = info.values.get('libraries')
print(f"Source code:\n{source}")
print(f"Libraries parsed: {libraries}")
print(f"Expected: ['foo']")
print(f"Actual: {libraries}")

# Test what's happening with different comment scenarios
print("\n" + "=" * 60)
print("Test 3: Various comment scenarios")
print("=" * 60)

# Test case 1: simple comment
source1 = """
# distutils: libraries = foo
"""
info1 = DistutilsInfo(source1)
print(f"Test 1 - No inline comment")
print(f"  Source: '# distutils: libraries = foo'")
print(f"  Result: {info1.values.get('libraries')}")

# Test case 2: inline comment with hash
source2 = """
# distutils: libraries = foo # comment here
"""
info2 = DistutilsInfo(source2)
print(f"\nTest 2 - Inline comment with hash")
print(f"  Source: '# distutils: libraries = foo # comment here'")
print(f"  Result: {info2.values.get('libraries')}")

# Test case 3: multiple libraries with inline comment
source3 = """
# distutils: libraries = foo bar # comment
"""
info3 = DistutilsInfo(source3)
print(f"\nTest 3 - Multiple libraries with inline comment")
print(f"  Source: '# distutils: libraries = foo bar # comment'")
print(f"  Result: {info3.values.get('libraries')}")

# Test case 4: list format with inline comment
source4 = """
# distutils: libraries = [foo, bar] # comment
"""
info4 = DistutilsInfo(source4)
print(f"\nTest 4 - List format with inline comment")
print(f"  Source: '# distutils: libraries = [foo, bar] # comment'")
print(f"  Result: {info4.values.get('libraries')}")

# Direct test of parse_list with hash
print("\n" + "=" * 60)
print("Test 4: Direct parse_list tests")
print("=" * 60)

test_cases = [
    "foo",
    "foo bar",
    "foo # bar",
    "# foo",
    "#",
    "# 0",
    "[foo, bar]",
    "[foo, bar] # comment",
]

for test in test_cases:
    result = parse_list(test)
    print(f"parse_list('{test}'): {result}")