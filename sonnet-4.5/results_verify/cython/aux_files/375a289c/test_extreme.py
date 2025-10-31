import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import time
from Cython.Compiler.StringEncoding import split_string_literal

# Test with extreme cases
print("Testing extreme cases:")
print("-" * 60)

# Test 1: 10000 backslashes with limit=10
s = '\\' * 10000
limit = 10

print(f"Test 1: {len(s)} backslashes with limit={limit}")
start = time.time()
result = split_string_literal(s, limit=limit)
elapsed = time.time() - start
print(f"Time taken: {elapsed:.4f} seconds")

# Verify correctness
rejoined = result.replace('""', '')
assert rejoined == s, f"Rejoined string doesn't match"

# Test 2: 50000 backslashes with limit=5
s = '\\' * 50000
limit = 5

print(f"\nTest 2: {len(s)} backslashes with limit={limit}")
start = time.time()
result = split_string_literal(s, limit=limit)
elapsed = time.time() - start
print(f"Time taken: {elapsed:.4f} seconds")

# Verify correctness
rejoined = result.replace('""', '')
assert rejoined == s, f"Rejoined string doesn't match"