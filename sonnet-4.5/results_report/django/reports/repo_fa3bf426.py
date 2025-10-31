#!/usr/bin/env python3
"""
Minimal reproduction of the handle_extensions bug that produces invalid '.' extension.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from django.core.management.utils import handle_extensions

print("Test case 1: Empty string")
result = handle_extensions([''])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 2: Double comma")
result = handle_extensions(['html,,css'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 3: Trailing comma")
result = handle_extensions(['html,'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 4: Leading comma")
result = handle_extensions([',html'])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("Test case 5: Just a comma")
result = handle_extensions([','])
print(f"Result: {result}")
assert '.' in result
print("✓ Confirmed: '.' is in the result\n")

print("All test cases confirm the bug: handle_extensions produces invalid '.' extension")