#!/usr/bin/env python3
"""Minimal reproduction of the django.utils.http.is_same_domain bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

# Test 1: An uppercase domain doesn't match itself
print("Test 1: is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')")
result1 = is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')
print(f"Result: {result1}")
print(f"Expected: True")
print(f"PASS" if result1 == True else f"FAIL: Domain doesn't match itself when uppercase\n")

# Test 2: Mixed-case domain doesn't match lowercase version
print("Test 2: is_same_domain('Example.COM', 'example.com')")
result2 = is_same_domain('Example.COM', 'example.com')
print(f"Result: {result2}")
print(f"Expected: True (DNS is case-insensitive)")
print(f"PASS" if result2 == True else f"FAIL: Mixed-case domain doesn't match lowercase version\n")

# Test 3: Uppercase subdomain doesn't match lowercase wildcard pattern
print("Test 3: is_same_domain('FOO.EXAMPLE.COM', '.example.com')")
result3 = is_same_domain('FOO.EXAMPLE.COM', '.example.com')
print(f"Result: {result3}")
print(f"Expected: True (should match wildcard pattern)")
print(f"PASS" if result3 == True else f"FAIL: Uppercase subdomain doesn't match lowercase wildcard\n")

# Test 4: Demonstrate asymmetry
print("Test 4: Asymmetric behavior")
print("  is_same_domain('example.com', 'EXAMPLE.COM'):", is_same_domain('example.com', 'EXAMPLE.COM'))
print("  is_same_domain('EXAMPLE.COM', 'example.com'):", is_same_domain('EXAMPLE.COM', 'example.com'))
print("FAIL: Function behaves asymmetrically with case\n")

# Test 5: Simple uppercase letter test
print("Test 5: is_same_domain('A', 'A')")
result5 = is_same_domain('A', 'A')
print(f"Result: {result5}")
print(f"Expected: True")
print(f"PASS" if result5 == True else f"FAIL: Single uppercase letter doesn't match itself")