#!/usr/bin/env python3
"""Reproduce the reported bug about shared mutable operators dict in Django dummy backend"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.backends.dummy.base import DatabaseWrapper

# Test 1: Check if operators is the same object between instances
wrapper1 = DatabaseWrapper({})
wrapper2 = DatabaseWrapper({})

print("Test 1: Check if operators dict is shared between instances")
print(f"wrapper1.operators id: {id(wrapper1.operators)}")
print(f"wrapper2.operators id: {id(wrapper2.operators)}")
print(f"Same object? {wrapper1.operators is wrapper2.operators}")
print()

# Test 2: Check if modifications affect other instances
print("Test 2: Check if modifying one instance affects the other")
print(f"wrapper1.operators before: {wrapper1.operators}")
print(f"wrapper2.operators before: {wrapper2.operators}")

wrapper1.operators['CUSTOM'] = 'custom_value'

print(f"After adding 'CUSTOM' to wrapper1:")
print(f"wrapper1.operators: {wrapper1.operators}")
print(f"wrapper2.operators: {wrapper2.operators}")
print(f"'CUSTOM' in wrapper2.operators: {'CUSTOM' in wrapper2.operators}")
print()

# Test 3: Create a third instance to verify it also gets the modification
wrapper3 = DatabaseWrapper({})
print("Test 3: Create a third instance after modification")
print(f"wrapper3.operators: {wrapper3.operators}")
print(f"'CUSTOM' in wrapper3.operators: {'CUSTOM' in wrapper3.operators}")