#!/usr/bin/env python3
"""
Minimal reproduction of the header injection vulnerability in
django.core.mail.message.forbid_multi_line_headers
"""

import sys
import os

# Add Django to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.mail.message import forbid_multi_line_headers

# The problematic input that causes the function to return a value containing newlines
name = 'X-Custom-Header'
val = '0\x0c\x80'
encoding = 'utf-8'

print("=" * 60)
print("REPRODUCING HEADER INJECTION VULNERABILITY")
print("=" * 60)
print()
print(f"Function: forbid_multi_line_headers")
print(f"Purpose: 'Forbid multi-line headers to prevent header injection'")
print()
print("INPUT:")
print(f"  name: {repr(name)}")
print(f"  val: {repr(val)}")
print(f"  encoding: {repr(encoding)}")
print()

# Call the function - it should prevent newlines but doesn't
result_name, result_val = forbid_multi_line_headers(name, val, encoding)

print("OUTPUT:")
print(f"  result_name: {repr(result_name)}")
print(f"  result_val: {repr(result_val)}")
print()

# Check if the output contains newlines
contains_newline = '\n' in result_val
contains_carriage_return = '\r' in result_val

print("ANALYSIS:")
print(f"  Contains \\n (newline): {contains_newline}")
print(f"  Contains \\r (carriage return): {contains_carriage_return}")
print()

if contains_newline or contains_carriage_return:
    print("❌ VULNERABILITY CONFIRMED!")
    print("The function returned a value with newlines, violating its")
    print("documented purpose of preventing header injection.")
    print()
    print("SECURITY IMPACT:")
    print("  - This allows header injection attacks")
    print("  - Attackers can inject additional email headers")
    print("  - Could lead to email spoofing and phishing")
else:
    print("✓ No vulnerability detected with this input")

print()
print("DETAILED OUTPUT BREAKDOWN:")
# Show the raw bytes to make the newline visible
print(f"  Raw bytes: {result_val.encode('utf-8')}")
# Show character-by-character
print("  Character breakdown:")
for i, char in enumerate(result_val):
    if char == '\n':
        print(f"    [{i}]: '\\n' (NEWLINE - SECURITY ISSUE)")
    elif char == '\r':
        print(f"    [{i}]: '\\r' (CARRIAGE RETURN - SECURITY ISSUE)")
    else:
        print(f"    [{i}]: {repr(char)}")