#!/usr/bin/env python3
"""Minimal reproduction of the cosine_similarity ZeroDivisionError bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

# Test case 1: Zero vector as first argument
print("Test 1: Zero vector as first argument")
print("Input: a = [0.0, 0.0, 0.0], b = [1.0, 2.0, 3.0]")
a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"Error: ZeroDivisionError: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: Zero vector as second argument
print("Test 2: Zero vector as second argument")
print("Input: a = [1.0, 2.0, 3.0], b = [0.0, 0.0, 0.0]")
a = [1.0, 2.0, 3.0]
b = [0.0, 0.0, 0.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"Error: ZeroDivisionError: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: Both vectors are zero
print("Test 3: Both vectors are zero")
print("Input: a = [0.0, 0.0, 0.0], b = [0.0, 0.0, 0.0]")
a = [0.0, 0.0, 0.0]
b = [0.0, 0.0, 0.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"Error: ZeroDivisionError: {e}")

print("\n" + "="*50 + "\n")

# Test case 4: Normal case (should work fine)
print("Test 4: Normal case with non-zero vectors")
print("Input: a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0]")
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")

    # Manual verification
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    expected = dot_product / (magnitude_a * magnitude_b)
    print(f"Manual calculation: {expected}")
    print(f"Results match: {abs(result - expected) < 1e-10}")
except ZeroDivisionError as e:
    print(f"Error: ZeroDivisionError: {e}")