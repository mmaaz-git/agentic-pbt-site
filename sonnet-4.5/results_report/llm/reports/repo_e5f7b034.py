import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
cosine_similarity = llm.cosine_similarity

# Test case with zero vector
a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

print(f"Testing cosine_similarity with:")
print(f"a = {a}")
print(f"b = {b}")
print()

try:
    result = cosine_similarity(a, b)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")