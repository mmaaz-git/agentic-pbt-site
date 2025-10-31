#!/usr/bin/env python3
"""Test to see how the execution handles multiple else clauses"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test with multiple else clauses
content = """
{{if x}}
true_branch
{{else}}
first_else
{{else}}
second_else
{{endif}}
"""

template = Template(content)

print("Parsed structure shows both else clauses:")
print(template._parsed)
print("\nNote: The structure contains THREE parts:")
print("1. ('if', ..., 'x', ['true_branch\\n'])")
print("2. ('else', ..., None, ['first_else\\n'])")
print("3. ('else', ..., None, ['second_else\\n'])")

print("\nExecution results:")
print("With x=True:", repr(template.substitute({'x': True})))
print("With x=False:", repr(template.substitute({'x': False})))

# Test with elif after else
content2 = """
{{if x}}
if_branch
{{else}}
else_branch
{{elif y}}
elif_branch
{{endif}}
"""

template2 = Template(content2)

print("\n\nFor elif after else:")
print("Parsed structure:")
print(template2._parsed)

print("\nExecution results:")
print("With x=False, y=True:", repr(template2.substitute({'x': False, 'y': True})))
print("With x=False, y=False:", repr(template2.substitute({'x': False, 'y': False})))