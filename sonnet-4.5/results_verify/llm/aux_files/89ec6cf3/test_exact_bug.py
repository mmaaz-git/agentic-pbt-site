import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

# The exact failing input from the bug report
a = [1, 0]
b = [1]

try:
    result = llm.cosine_similarity(a, b)
    print(f"SUCCESS: Function returned {result} for mismatched vector lengths")
    print(f"  a = {a} (length {len(a)})")
    print(f"  b = {b} (length {len(b)})")
    print("\nThis demonstrates the bug - function silently returns incorrect result")
except Exception as e:
    print(f"FAILED: Got exception: {e}")
    print("This would mean the function correctly rejects mismatched lengths")