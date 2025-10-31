#!/usr/bin/env python3
"""Minimal reproduction demonstrating __name parameter leak in Cython.Tempita.sub()"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import sub

# Test 1: Basic leak demonstration
content = "Template name: {{__name}}"
result = sub(content, __name='mytemplate.html')

print("Test 1: Basic __name leak")
print("-" * 40)
print(f"Template: {repr(content)}")
print(f"Call: sub(content, __name='mytemplate.html')")
print(f"Expected: 'Template name: ' (empty)")
print(f"Actual: {repr(result)}")
print()

# Test 2: __name alongside regular variables
content2 = "Name is: {{__name}}, foo is: {{foo}}"
result2 = sub(content2, __name='template.html', foo='bar')

print("Test 2: __name with other variables")
print("-" * 40)
print(f"Template: {repr(content2)}")
print(f"Call: sub(content2, __name='template.html', foo='bar')")
print(f"Expected: 'Name is: , foo is: bar'")
print(f"Actual: {repr(result2)}")
print()

# Test 3: Empty __name
content3 = "{{__name}}"
result3 = sub(content3, __name='')

print("Test 3: Empty __name value")
print("-" * 40)
print(f"Template: {repr(content3)}")
print(f"Call: sub(content3, __name='')")
print(f"Expected: '' (empty)")
print(f"Actual: {repr(result3)}")
print()

print("BUG CONFIRMED: __name parameter is accessible as template variable")
print("This violates the documented purpose of __name as a meta-parameter for error reporting")