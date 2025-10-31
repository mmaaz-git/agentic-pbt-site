#!/usr/bin/env python3
"""Test script to reproduce the reported bugs"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.dataframe.dask_expr._repartition import _clean_new_division_boundaries

print("Testing Bug 1: boundaries=[2, 0], frame_npartitions=1")
boundaries1 = [2, 0]
result1 = _clean_new_division_boundaries(boundaries1.copy(), 1)
print(f"  Input: {[2, 0]}, frame_npartitions=1")
print(f"  Output: {result1}")
print(f"  Expected by bug report: [0, 2, 1]")
print(f"  Is non-monotonic? {any(result1[i] > result1[i+1] for i in range(len(result1)-1))}")
print()

print("Testing Bug 2: boundaries=[0, 2], frame_npartitions=1")
boundaries2 = [0, 2]
result2 = _clean_new_division_boundaries(boundaries2.copy(), 1)
print(f"  Input: {[0, 2]}, frame_npartitions=1")
print(f"  Output: {result2}")
print(f"  Expected by bug report: [0, 2]")
print(f"  Last element equals frame_npartitions? {result2[-1] == 1}")
print()

print("Testing Bug 3: boundaries=[0, 5, 10], frame_npartitions=20")
boundaries3 = [0, 5, 10]
result3 = _clean_new_division_boundaries(boundaries3.copy(), 20)
print(f"  Input: {[0, 5, 10]}, frame_npartitions=20")
print(f"  Output: {result3}")
print(f"  Expected by bug report: [0, 5, 20]")
print(f"  Lost boundary 10? {10 not in result3}")
print()

# Test the property-based test
print("Testing general properties:")
test_cases = [
    ([2, 0], 1),
    ([0, 2], 1),
    ([0, 5, 10], 20),
    ([1, 3, 5], 10),
    ([5], 3),
    ([0, 1, 2, 3], 3),
]

for boundaries, frame_npartitions in test_cases:
    result = _clean_new_division_boundaries(boundaries.copy(), frame_npartitions)
    print(f"  Input: {boundaries}, frame_npartitions={frame_npartitions}")
    print(f"    Output: {result}")

    # Check properties
    props = []
    if result[0] != 0:
        props.append("first!=0")
    if result[-1] != frame_npartitions:
        props.append(f"last!=frame_npartitions({result[-1]}!={frame_npartitions})")
    if any(result[i] > result[i+1] for i in range(len(result)-1)):
        props.append("non-monotonic")

    if props:
        print(f"    Issues: {', '.join(props)}")
    else:
        print(f"    OK")