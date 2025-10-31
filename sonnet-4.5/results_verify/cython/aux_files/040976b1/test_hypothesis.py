#!/usr/bin/env python3
"""Test using hypothesis as described in bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template, TemplateError

def test_with_condition(condition):
    content = """
{{if x}}
a
{{else}}
b
{{else}}
c
{{endif}}
"""

    template = Template(content)
    # This should raise TemplateError but doesn't
    result = template.substitute({'x': condition})
    return result

# Test both conditions
print("Testing with x=True:")
result1 = test_with_condition(True)
print(f"Result: {repr(result1)}")

print("\nTesting with x=False:")
result2 = test_with_condition(False)
print(f"Result: {repr(result2)}")

print("\nConclusion: The template accepts multiple else clauses without raising an error.")
print("The second else clause is silently ignored.")