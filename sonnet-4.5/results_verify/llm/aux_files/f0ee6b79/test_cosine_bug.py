#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

print("Testing cosine_similarity with zero vector...")

# Test case from bug report
a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

print(f"a = {a}")
print(f"b = {b}")

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test the reverse case
print("\nTesting reverse (non-zero, zero)...")
a = [1.0, 2.0, 3.0]
b = [0.0, 0.0, 0.0]

print(f"a = {a}")
print(f"b = {b}")

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test both vectors as zero
print("\nTesting both vectors as zero...")
a = [0.0, 0.0, 0.0]
b = [0.0, 0.0, 0.0]

print(f"a = {a}")
print(f"b = {b}")

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test normal case to verify function works
print("\nTesting normal case...")
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

print(f"a = {a}")
print(f"b = {b}")

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")

    # Manually calculate to verify
    import math
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    expected = dot_product / (magnitude_a * magnitude_b)
    print(f"Expected (manual calculation): {expected}")
    print(f"Match: {abs(result - expected) < 1e-10}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")