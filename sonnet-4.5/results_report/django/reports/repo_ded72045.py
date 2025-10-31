#!/usr/bin/env python3
"""Minimal reproduction of the django.core.management.utils.handle_extensions bug."""

from django.core.management.utils import handle_extensions

print("Testing handle_extensions with various comma-separated inputs that contain empty strings:")
print()

# Test case 1: Double comma creates empty string
print("Test 1: handle_extensions(['py,,js'])")
result1 = handle_extensions(['py,,js'])
print(f"Result: {result1}")
print(f"Contains '.': {'.' in result1}")
print()

# Test case 2: Trailing comma
print("Test 2: handle_extensions(['py,'])")
result2 = handle_extensions(['py,'])
print(f"Result: {result2}")
print(f"Contains '.': {'.' in result2}")
print()

# Test case 3: Space between commas
print("Test 3: handle_extensions(['py, ,js'])")
result3 = handle_extensions(['py, ,js'])
print(f"Result: {result3}")
print(f"Contains '.': {'.' in result3}")
print()

# Test case 4: Leading comma
print("Test 4: handle_extensions([',py'])")
result4 = handle_extensions([',py'])
print(f"Result: {result4}")
print(f"Contains '.': {'.' in result4}")
print()

# Test case 5: Multiple consecutive commas
print("Test 5: handle_extensions(['py,,,js'])")
result5 = handle_extensions(['py,,,js'])
print(f"Result: {result5}")
print(f"Contains '.': {'.' in result5}")
print()

print("Summary:")
print("The function incorrectly returns '.' as a valid extension when empty strings")
print("are present in the comma-separated input. This happens because empty strings")
print("from split() are prefixed with '.' without checking if they are actually empty.")