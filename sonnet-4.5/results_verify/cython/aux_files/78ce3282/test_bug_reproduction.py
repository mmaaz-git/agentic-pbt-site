#!/usr/bin/env python3
"""Test to reproduce the source_to_lineno bug"""

from hypothesis import given, strategies as st, settings
import os

# First, let's test with the actual codefile
codefile = '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Debugger/Tests/codefile'

print("=" * 60)
print("Testing with actual codefile:")
print("=" * 60)

with open(codefile) as f:
    lines = list(f)
    source_to_lineno = {line.strip(): i for i, line in enumerate(lines, 1)}

# Find all occurrences of 'pass'
pass_occurrences = [i for i, line in enumerate(lines, 1) if line.strip() == 'pass']
print(f"'pass' appears on lines: {pass_occurrences}")
print(f"source_to_lineno['pass'] = {source_to_lineno.get('pass', 'NOT FOUND')}")

# Verify the bug
if 'pass' in source_to_lineno:
    assert source_to_lineno['pass'] == pass_occurrences[-1], \
        f"Dictionary maps to last occurrence: {source_to_lineno['pass']} == {pass_occurrences[-1]}"
    print(f"✓ Bug confirmed: 'pass' maps to last occurrence only (line {source_to_lineno['pass']})")
    print(f"  Lost mappings for lines: {pass_occurrences[:-1]}")

print("\n" + "=" * 60)
print("Testing with example from bug report:")
print("=" * 60)

test_file_content = """cpdef eggs():
    pass

cdef ham():
    pass

cdef class SomeClass(object):
    def spam(self):
        pass
"""

lines = test_file_content.strip().split('\n')
source_to_lineno = {line.strip(): i for i, line in enumerate(lines, 1)}

pass_occurrences = [i for i, line in enumerate(lines, 1) if line.strip() == 'pass']
print(f"'pass' appears on lines: {pass_occurrences}")
print(f"source_to_lineno['pass'] = {source_to_lineno['pass']}")

assert pass_occurrences == [2, 5, 9]
assert source_to_lineno['pass'] == 9
print("✓ Bug confirmed with example code")

print("\n" + "=" * 60)
print("Testing with property-based test:")
print("=" * 60)

@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=50))
@settings(max_examples=100)  # Reduced for quicker testing
def test_source_to_lineno_preserves_all_lines(lines):
    lines_with_duplicates = lines + [lines[0]]

    source_to_lineno = {line.strip(): i for i, line in enumerate(lines_with_duplicates, 1)}

    first_occurrence = 1
    last_occurrence = len(lines_with_duplicates)

    mapped_lineno = source_to_lineno[lines[0].strip()]

    assert mapped_lineno == first_occurrence or mapped_lineno == last_occurrence

    if len(lines) != len(set(line.strip() for line in lines)):
        assert mapped_lineno == last_occurrence, \
            "Dictionary comprehension maps duplicate keys to last value, losing earlier occurrences"

# Run the property-based test
try:
    test_source_to_lineno_preserves_all_lines()
    print("Property-based test passed (100 examples)")
except AssertionError as e:
    print(f"Property-based test failed: {e}")

print("\n" + "=" * 60)
print("Testing dictionary behavior with duplicates:")
print("=" * 60)

# Demonstrate the fundamental Python behavior
test_dict = {'key': 1, 'key': 2, 'key': 3}
print(f"Dict literal with duplicate keys: {test_dict}")

# Dictionary comprehension with duplicates
items = [('key', 1), ('key', 2), ('key', 3)]
comp_dict = {k: v for k, v in items}
print(f"Dict comprehension with duplicates: {comp_dict}")

# This is the exact pattern used in the code
lines_demo = ['line1', 'pass', 'line3', 'pass', 'line5', 'pass']
demo_dict = {line.strip(): i for i, line in enumerate(lines_demo, 1)}
print(f"Pattern from code: {demo_dict}")
print(f"'pass' maps to line {demo_dict['pass']} (should be 6, loses 2 and 4)")