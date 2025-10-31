#!/usr/bin/env python3
"""Minimal reproduction of the get_digit bug with negative numbers."""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from django.template import defaultfilters

# Test case 1: get_digit(-123, 4)
print("Test case 1: get_digit(-123, 4)")
print("Expected: Should return 0 or -123 (original value) based on documented behavior")
print("String representation: str(-123) = '-123' (4 characters)")
print("Accessing position -4: str(-123)[-4] = '-'")
print("Attempting to execute...")

try:
    result = defaultfilters.get_digit(-123, 4)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: get_digit(-1, 2)
print("Test case 2: get_digit(-1, 2)")
print("Expected: Should return 0 or -1 (original value) based on documented behavior")
print("String representation: str(-1) = '-1' (2 characters)")
print("Accessing position -2: str(-1)[-2] = '-'")
print("Attempting to execute...")

try:
    result = defaultfilters.get_digit(-1, 2)
    print(f"Result: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")