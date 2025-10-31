import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import time
from Cython.Compiler.StringEncoding import split_string_literal

# Test with varying sizes to see the performance degradation
test_cases = [
    (50, 10),
    (100, 10),
    (150, 10),
    (200, 10),
    (250, 10),
    (300, 10),
]

print("Testing split_string_literal performance with backslashes:")
print("-" * 60)

for num_backslashes, limit in test_cases:
    s = '\\' * num_backslashes

    start = time.time()
    result = split_string_literal(s, limit=limit)
    elapsed = time.time() - start

    print(f"Backslashes: {num_backslashes:3d}, Limit: {limit:2d}, Time: {elapsed:.4f}s")

    # Verify correctness
    rejoined = result.replace('""', '')
    assert rejoined == s, f"Rejoined string doesn't match: expected {len(s)} chars, got {len(rejoined)}"

    # Stop if it's taking too long
    if elapsed > 5:
        print("Stopping test - performance is too slow!")
        break