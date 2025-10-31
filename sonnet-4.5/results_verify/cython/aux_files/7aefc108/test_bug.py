#!/usr/bin/env python3
"""Test the reported bug about Template.substitute mutating input dictionary"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

print("=== Test 1: Basic mutation test ===")
template = Template("{{x}}", name="test.tmpl")
input_dict = {'x': 42}

print(f"Before: {input_dict}")
print(f"Keys before: {list(input_dict.keys())}")
result = template.substitute(input_dict)
print(f"After:  {input_dict}")
print(f"Keys after: {list(input_dict.keys())}")
print(f"Result: {result}")

if '__template_name__' in input_dict:
    print("\nBUG CONFIRMED: Input dictionary was mutated!")
    print(f"Added key '__template_name__' with value: {input_dict['__template_name__']}")
else:
    print("\nNo mutation detected")

print("\n=== Test 2: Multiple substitutions ===")
template2 = Template("{{y}}", name="another.tmpl")
dict2 = {'y': 100}
print(f"Dict before first substitution: {dict2}")
result1 = template2.substitute(dict2)
print(f"Dict after first substitution: {dict2}")
result2 = template2.substitute(dict2)  # Reuse same dict
print(f"Dict after second substitution: {dict2}")

print("\n=== Test 3: Check if namespace is also mutated ===")
template3 = Template("{{z}}", namespace={'default': 'value'})
dict3 = {'z': 'test'}
print(f"Dict before: {dict3}")
result3 = template3.substitute(dict3)
print(f"Dict after: {dict3}")
print(f"'default' in dict3: {'default' in dict3}")

print("\n=== Test 4: Property-based test from bug report ===")
from hypothesis import given, strategies as st, settings
import string

@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=1, max_size=5
))
@settings(max_examples=10)
def test_substitute_does_not_mutate_input(input_dict):
    original_keys = set(input_dict.keys())

    content = "{{x}}" if 'x' in input_dict else "test"
    template = Template(content)
    result = template.substitute(input_dict)

    new_keys = set(input_dict.keys())
    added_keys = new_keys - original_keys

    if added_keys:
        print(f"FAILED with input {input_dict}: Added keys {added_keys}")
        return False
    return True

# Run the property test
print("Running property-based test...")
try:
    test_substitute_does_not_mutate_input()
    print("Property test completed")
except Exception as e:
    print(f"Property test failed: {e}")