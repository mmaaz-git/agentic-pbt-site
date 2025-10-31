#!/usr/bin/env python3
"""
Minimal reproduction of the dask.sizeof dict non-determinism bug.
Demonstrates that sizeof(dict) returns different values on repeated calls
when the dict has more than 10 items.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')
from dask.sizeof import sizeof

# Create a dictionary with 11 items (triggers the bug)
d = {'': 0, '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '00': 0, '000': 0, '0000': 0, '00000': 0, '000000': 0}

# Call sizeof multiple times on the same dict
results = [sizeof(d) for _ in range(20)]

print(f"Dictionary: {d}")
print(f"Number of items: {len(d)}")
print(f"\nResults from 20 calls to sizeof(d):")
print(results)
print(f"\nUnique values: {sorted(set(results))}")
print(f"Min value: {min(results)}")
print(f"Max value: {max(results)}")
print(f"Range: {max(results) - min(results)} bytes difference")

# Show that the issue is due to > 10 items
print("\n--- Comparison with 10 items (should be deterministic) ---")
d_10 = {str(i): i for i in range(10)}
results_10 = [sizeof(d_10) for _ in range(20)]
print(f"Dict with 10 items - Results: {sorted(set(results_10))}")
print(f"Is deterministic: {len(set(results_10)) == 1}")

print("\n--- Comparison with 11 items (non-deterministic) ---")
d_11 = {str(i): i for i in range(11)}
results_11 = [sizeof(d_11) for _ in range(20)]
print(f"Dict with 11 items - Results: {sorted(set(results_11))}")
print(f"Is deterministic: {len(set(results_11)) == 1}")