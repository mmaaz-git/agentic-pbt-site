#!/usr/bin/env python3
"""Test using synchronous scheduler to avoid multiprocessing issues"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask
import dask.bag as db
from operator import add

# Use synchronous scheduler
dask.config.set(scheduler='synchronous')

print("Testing with empty first partition")
print("=" * 60)

# Create bag with empty first partition
dsk = {
    ('test', 0): [],
    ('test', 1): [1, 2, 3]
}
bag = db.Bag(dsk, 'test', 2)

print(f"Partitions: {[dsk[('test', i)] for i in range(2)]}")

# Now test accumulate
result = bag.accumulate(add, initial=0)
computed = result.compute()

print(f"Result: {computed}")
print(f"Expected: [0, 1, 3, 6]")

# The bug would cause wrong behavior here
# Let's trace what happens:
print("\n" + "=" * 60)
print("Tracing the execution:")

from dask.bag.core import accumulate_part
from tlz import first, second

# First partition (empty)
part1 = accumulate_part(add, [], 0, is_first=True)
print(f"Part 1: accumulate_part(add, [], 0, is_first=True)")
print(f"  Returns: {part1}")
print(f"  Length: {len(part1)}")

if len(part1) == 3:
    print("  BUG: Returns 3-tuple instead of 2-tuple!")
    print(f"  first(part1) = {first(part1)}")
    print(f"  second(part1) = {second(part1)}")
    carry = second(part1)
else:
    print(f"  first(part1) = {first(part1)}")
    print(f"  second(part1) = {second(part1)}")
    carry = second(part1)

# Second partition
part2 = accumulate_part(add, [1, 2, 3], carry, is_first=False)
print(f"\nPart 2: accumulate_part(add, [1, 2, 3], {carry}, is_first=False)")
print(f"  Returns: {part2}")
print(f"  first(part2) = {first(part2)}")

# Combined result
final_result = list(first(part1)) + list(first(part2))
print(f"\nCombined result: {final_result}")

print("\n" + "=" * 60)
print("Despite the bug, it seems to work because:")
print("1. When res=[0] (non-empty), the 3-tuple is ([0], 0, 0)")
print("2. second() on ([0], 0, 0) returns 0 (the middle element)")
print("3. This happens to be the correct carry value!")
print("\nSo the bug exists but doesn't cause incorrect results in this case.")