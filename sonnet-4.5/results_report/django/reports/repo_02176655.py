#!/usr/bin/env python3
"""Minimal reproduction of django.utils.http.is_same_domain case sensitivity bug"""

# Add Django environment to path
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

# Test basic uppercase domains - should return True but returns False
print("Test 1: is_same_domain('A', 'A') =", is_same_domain('A', 'A'))
print("Expected: True, Got:", is_same_domain('A', 'A'))
print()

# Test mixed case domain matching - should return True but returns False
print("Test 2: is_same_domain('example.COM', 'EXAMPLE.com') =", is_same_domain('example.COM', 'EXAMPLE.com'))
print("Expected: True, Got:", is_same_domain('example.COM', 'EXAMPLE.com'))
print()

print("Test 3: is_same_domain('Example.Com', 'example.com') =", is_same_domain('Example.Com', 'example.com'))
print("Expected: True, Got:", is_same_domain('Example.Com', 'example.com'))
print()

# Show asymmetric behavior
print("Asymmetric behavior demonstration:")
print("  is_same_domain('a', 'A') =", is_same_domain('a', 'A'))  # Returns True
print("  is_same_domain('A', 'a') =", is_same_domain('A', 'a'))  # Returns False
print("This asymmetry shows only pattern is lowercased, not host")
print()

# Test subdomain matching with case sensitivity
print("Subdomain matching:")
print("  is_same_domain('sub.EXAMPLE.com', '.example.com') =", is_same_domain('sub.EXAMPLE.com', '.example.com'))
print("  is_same_domain('SUB.example.com', '.EXAMPLE.COM') =", is_same_domain('SUB.example.com', '.EXAMPLE.COM'))
print()

# Real-world scenario - HTTP Host headers can have any case
print("Real-world scenario (HTTP Host header with various cases):")
host_variations = ['Example.Com', 'EXAMPLE.COM', 'example.com', 'ExAmPlE.cOm']
pattern = 'example.com'
print(f"Pattern: {pattern}")
for host in host_variations:
    result = is_same_domain(host, pattern)
    print(f"  is_same_domain('{host}', '{pattern}') = {result}")