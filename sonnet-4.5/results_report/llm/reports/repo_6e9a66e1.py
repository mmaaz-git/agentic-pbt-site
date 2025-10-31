import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

# Test case with zero vector
a = [1.0, 2.0, 3.0]
b = [0.0, 0.0, 0.0]

print(f"Testing llm.cosine_similarity with:")
print(f"  a = {a}")
print(f"  b = {b}")
print()

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()