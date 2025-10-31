#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

print(f"Trying cosine_similarity with a={a} and b={b}")
try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Also try with both vectors being zero
a2 = [0.0, 0.0]
b2 = [0.0, 0.0]
print(f"\nTrying cosine_similarity with a={a2} and b={b2}")
try:
    result = llm.cosine_similarity(a2, b2)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")