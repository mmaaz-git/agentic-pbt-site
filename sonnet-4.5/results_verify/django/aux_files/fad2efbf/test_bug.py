#!/usr/bin/env python3
"""Test script to reproduce the is_same_domain bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.http import is_same_domain

# Test 1: The hypothesis test case
print("Test 1 - Hypothesis test case:")
print(f"is_same_domain('A', 'A') = {is_same_domain('A', 'A')}")
print(f"Expected: True, Got: {is_same_domain('A', 'A')}")
print()

# Test 2: Examples from bug report
print("Test 2 - Examples from bug report:")
result1 = is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM')
print(f"is_same_domain('EXAMPLE.COM', 'EXAMPLE.COM') = {result1}")
print(f"Expected: True (domains should match themselves), Got: {result1}")
print()

result2 = is_same_domain('Example.COM', 'example.com')
print(f"is_same_domain('Example.COM', 'example.com') = {result2}")
print(f"Expected: True (case-insensitive match), Got: {result2}")
print()

result3 = is_same_domain('FOO.EXAMPLE.COM', '.example.com')
print(f"is_same_domain('FOO.EXAMPLE.COM', '.example.com') = {result3}")
print(f"Expected: True (subdomain match), Got: {result3}")
print()

# Additional tests to understand the behavior
print("Test 3 - Additional tests:")
print(f"is_same_domain('example.com', 'EXAMPLE.COM') = {is_same_domain('example.com', 'EXAMPLE.COM')}")
print(f"is_same_domain('example.com', 'example.com') = {is_same_domain('example.com', 'example.com')}")
print(f"is_same_domain('EXAMPLE.COM', 'example.com') = {is_same_domain('EXAMPLE.COM', 'example.com')}")
print()

# Test with wildcard patterns
print("Test 4 - Wildcard patterns:")
print(f"is_same_domain('foo.example.com', '.EXAMPLE.COM') = {is_same_domain('foo.example.com', '.EXAMPLE.COM')}")
print(f"is_same_domain('FOO.EXAMPLE.COM', '.EXAMPLE.COM') = {is_same_domain('FOO.EXAMPLE.COM', '.EXAMPLE.COM')}")
print(f"is_same_domain('foo.example.com', '.example.com') = {is_same_domain('foo.example.com', '.example.com')}")