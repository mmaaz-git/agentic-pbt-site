#!/usr/bin/env python3
"""Analyze the truncate_name implementation logic"""

from django.db.backends.utils import truncate_name, split_identifier

# Test to understand the logic
identifier = 'SCHEMA"."VERYLONGTABLENAME'
namespace, name = split_identifier(identifier)

print("Original identifier:", identifier)
print("After split_identifier:")
print(f"  namespace: '{namespace}'")
print(f"  name: '{name}'")
print()

# The current implementation checks: len(name) <= length
length = 20
print(f"Length limit: {length}")
print(f"len(name): {len(name)}")
print(f"len(name) <= length: {len(name) <= length}")
print(f"Current logic would return original? {len(name) <= length}")
print()

# What actually gets returned
print("Full identifier components:")
print(f"  namespace part: '{namespace}\"'.\"'")
print(f"  name part: '{name}'")
print(f"  Combined length: {len(namespace) + 3 + len(name)}")
print()

# Test when name needs truncation
identifier2 = 'SCHEMA"."VERYLONGTABLENAMETHATEXCEEDSLIMIT'
namespace2, name2 = split_identifier(identifier2)
length2 = 20

print("Second test:")
print(f"  Identifier: {identifier2}")
print(f"  namespace: '{namespace2}'")
print(f"  name: '{name2}'")
print(f"  name length: {len(name2)}")
print(f"  Length limit: {length2}")

result = truncate_name(identifier2, length=length2)
print(f"  Result: {result}")
print(f"  Result length: {len(result.strip('\"'))}")