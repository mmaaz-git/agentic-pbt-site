#!/usr/bin/env python3
"""Test to reproduce the FastAPI get_path_param_names bug"""

from fastapi.utils import get_path_param_names

# Test 1: Basic reproduction
print("Test 1: Basic reproduction")
result = get_path_param_names("/users/{}/posts")
print(f"Result: {result}")
print(f"Empty string in result: {'' in result}")
assert '' in result, "Bug NOT reproduced - empty string should be in result"
print("✓ Bug reproduced: Empty string found in result\n")

# Test 2: Multiple empty parameters
print("Test 2: Multiple empty parameters")
result2 = get_path_param_names("/api/{}/items/{}/details")
print(f"Result: {result2}")
print(f"Empty string in result: {'' in result2}")
print(f"Number of empty strings: {list(result2).count('')}")
print()

# Test 3: Mixed empty and non-empty parameters
print("Test 3: Mixed empty and non-empty parameters")
result3 = get_path_param_names("/users/{user_id}/posts/{}/comments/{comment_id}")
print(f"Result: {result3}")
print(f"Contains empty string: {'' in result3}")
print()

# Test 4: Normal case (no empty parameters)
print("Test 4: Normal case (no empty parameters)")
result4 = get_path_param_names("/users/{user_id}/posts/{post_id}")
print(f"Result: {result4}")
print(f"Contains empty string: {'' in result4}")