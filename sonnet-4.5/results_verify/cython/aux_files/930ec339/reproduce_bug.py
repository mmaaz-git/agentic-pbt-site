#!/usr/bin/env python3
"""Reproduce the Cython.Tempita Template.substitute namespace override bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case from bug report
template = Template('{{x}}', namespace={'x': 'namespace_value'})
result = template.substitute({'x': 'substitute_value'})

print(f"Result: {result}")
print(f"Expected: substitute_value")
print(f"Actual: {result}")

# Additional test cases
print("\n--- Additional test cases ---")

# Test with integer values
template2 = Template('{{num}}', namespace={'num': 100})
result2 = template2.substitute({'num': 200})
print(f"\nInteger test - Result: {result2}")
print(f"Expected: 200")
print(f"Actual: {result2}")

# Test with multiple variables
template3 = Template('{{x}} {{y}}', namespace={'x': 'ns_x', 'y': 'ns_y'})
result3 = template3.substitute({'x': 'sub_x', 'y': 'sub_y'})
print(f"\nMultiple vars - Result: {result3}")
print(f"Expected: sub_x sub_y")
print(f"Actual: {result3}")

# Test with partial override
template4 = Template('{{x}} {{y}}', namespace={'x': 'ns_x', 'y': 'ns_y'})
result4 = template4.substitute({'x': 'sub_x'})
print(f"\nPartial override - Result: {result4}")
print(f"Expected: sub_x ns_y")
print(f"Actual: {result4}")