#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

print("Testing Bug 1: Non-monotonic boundaries")
boundaries1 = [2, 0]
print(f"Input: boundaries={boundaries1}, frame_npartitions=1")
result1 = _clean_new_division_boundaries(boundaries1.copy(), 1)
print(f"Output: {result1}")
print(f"Is non-monotonic? {result1[1] > result1[2]} (element at index 1 > element at index 2)")
print()

print("Testing Bug 2: Last boundary != frame_npartitions")
boundaries2 = [0, 2]
print(f"Input: boundaries={boundaries2}, frame_npartitions=1")
result2 = _clean_new_division_boundaries(boundaries2.copy(), 1)
print(f"Output: {result2}")
print(f"Last element equals frame_npartitions? {result2[-1] == 1} (last={result2[-1]}, expected=1)")
print()

print("Testing Bug 3: Lost intermediate boundary")
boundaries3 = [0, 5, 10]
print(f"Input: boundaries={boundaries3}, frame_npartitions=20")
result3 = _clean_new_division_boundaries(boundaries3.copy(), 20)
print(f"Output: {result3}")
print(f"Original boundary 10 lost? {10 not in result3}")
print(f"Original length: {len([0, 5, 10])}, Result length: {len(result3)}")