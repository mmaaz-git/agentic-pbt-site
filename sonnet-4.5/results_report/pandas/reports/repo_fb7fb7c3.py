import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers.objects import FixedWindowIndexer

# Create an indexer with window_size=0
indexer = FixedWindowIndexer(window_size=0)

# Get window bounds with problematic parameters
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"Window size: 0")
print(f"Num values: 2")
print(f"Closed: 'neither'")
print(f"Start: {start}")
print(f"End: {end}")
print()

# Check if the invariant holds
for i in range(len(start)):
    print(f"Window {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")

# Show which windows violate the invariant
violations = []
for i in range(len(start)):
    if start[i] > end[i]:
        violations.append(i)

if violations:
    print(f"\nInvariant violations found at indices: {violations}")
    print("This means window start > window end, which is invalid for a window range.")