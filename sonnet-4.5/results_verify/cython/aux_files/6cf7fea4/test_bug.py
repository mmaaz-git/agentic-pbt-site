#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

print("=== Testing Template.substitute() mutation bug ===\n")

# Test 1: Basic mutation test
print("Test 1: Basic dictionary mutation")
user_dict = {'x': 'value', 'y': 42}
print(f"Before: {user_dict}")

template = Template('{{x}}', namespace={'z': 100})
result = template.substitute(user_dict)

print(f"After:  {user_dict}")
print(f"Result: {result}")
print(f"Added keys: {set(user_dict.keys()) - {'x', 'y'}}\n")

# Test 2: Test with empty namespace
print("Test 2: Template with no namespace")
user_dict2 = {'a': 1, 'b': 2}
original_dict2 = user_dict2.copy()
print(f"Before: {user_dict2}")

template2 = Template('{{a}} and {{b}}')
result2 = template2.substitute(user_dict2)

print(f"After:  {user_dict2}")
print(f"Result: {result2}")
print(f"Dict changed: {user_dict2 != original_dict2}")
print(f"Added keys: {set(user_dict2.keys()) - set(original_dict2.keys())}\n")

# Test 3: Property-based test from the bug report
print("Test 3: Property-based test scenario")
from hypothesis import given, strategies as st
import string

@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=1, max_size=3
))
def test_substitute_does_not_mutate_input(user_vars):
    if not user_vars:
        user_vars = {'x': 1}

    var_to_use = list(user_vars.keys())[0]
    template = Template(f'{{{{{var_to_use}}}}}', namespace={'other': 999})
    original_vars = user_vars.copy()

    template.substitute(user_vars)

    assert user_vars == original_vars, f"Dictionary was mutated! Original: {original_vars}, After: {user_vars}"

# Run a few examples manually (without hypothesis decorator)
print("Running manual test with sample inputs...")
test_examples = [
    {'x': 1},
    {'foo': 42, 'bar': 100},
    {'test': -5}
]

for example in test_examples:
    original = example.copy()
    var_to_use = list(example.keys())[0]
    template = Template(f'{{{{{var_to_use}}}}}', namespace={'other': 999})

    template.substitute(example)

    if example != original:
        print(f"  Dictionary was mutated!")
        print(f"    Original: {original}")
        print(f"    After:    {example}")
    else:
        print(f"  OK: {original} unchanged")

print("\n=== Comparison with string.Template ===")
from string import Template as StdTemplate

print("Testing Python's string.Template for comparison:")
std_dict = {'name': 'World'}
print(f"Before: {std_dict}")

std_template = StdTemplate("Hello $name")
std_result = std_template.substitute(std_dict)

print(f"After:  {std_dict}")
print(f"Result: {std_result}")
print(f"Dict modified: {len(std_dict) != 1}")