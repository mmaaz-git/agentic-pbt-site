import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers.objects import FixedWindowIndexer

indexer = FixedWindowIndexer(window_size=0)
start, end = indexer.get_window_bounds(num_values=2, closed='neither')

print(f"Window size: 0")
print(f"Num values: 2")
print(f"Closed: 'neither'")
print(f"Start: {start}")
print(f"End: {end}")

for i in range(len(start)):
    print(f"Window {i}: start={start[i]}, end={end[i]}, valid={start[i] <= end[i]}")