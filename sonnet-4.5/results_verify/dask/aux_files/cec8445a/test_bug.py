#!/usr/bin/env python3
"""Test the truncate_name bug report"""

from django.db.backends.utils import truncate_name

# Test 1: The exact example from the bug report
print("Test 1: Bug report example")
identifier = 'SCHEMA"."VERYLONGTABLENAME'
length = 20

result = truncate_name(identifier, length=length)
print(f"Input: {identifier}")
print(f"Length limit: {length}")
print(f"Result: {result}")
print(f"Result length: {len(result.strip('\"'))}")
print(f"Expected max length: {length}")
print(f"Passes length check: {len(result.strip('\"')) <= length}")
print()

# Test 2: Without namespace - should work correctly
print("Test 2: Without namespace")
identifier2 = "VERYLONGTABLENAME"
length2 = 10
result2 = truncate_name(identifier2, length=length2)
print(f"Input: {identifier2}")
print(f"Length limit: {length2}")
print(f"Result: {result2}")
print(f"Result length: {len(result2)}")
print(f"Passes length check: {len(result2) <= length2}")
print()

# Test 3: With namespace where table name is already short
print("Test 3: Short table name with namespace")
identifier3 = 'SCHEMA"."SHORT'
length3 = 20
result3 = truncate_name(identifier3, length=length3)
print(f"Input: {identifier3}")
print(f"Length limit: {length3}")
print(f"Result: {result3}")
print(f"Result length: {len(result3.strip('\"'))}")
print(f"Passes length check: {len(result3.strip('\"')) <= length3}")
print()

# Test 4: Edge case with very long namespace
print("Test 4: Long namespace")
identifier4 = 'VERYLONGSCHEMANAME"."VERYLONGTABLENAME'
length4 = 30
result4 = truncate_name(identifier4, length=length4)
print(f"Input: {identifier4}")
print(f"Length limit: {length4}")
print(f"Result: {result4}")
print(f"Result length: {len(result4.strip('\"'))}")
print(f"Passes length check: {len(result4.strip('\"')) <= length4}")
print()

# Test 5: Test with None length (should return original)
print("Test 5: None length parameter")
identifier5 = 'SCHEMA"."VERYLONGTABLENAME'
result5 = truncate_name(identifier5, length=None)
print(f"Input: {identifier5}")
print(f"Result: {result5}")
print(f"Should be identical: {identifier5 == result5}")