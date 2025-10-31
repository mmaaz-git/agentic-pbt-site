from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-9_223_372_036_854_775_809)
start, end = indexer.get_window_bounds(num_values=1)
print(f"start: {start}, end: {end}")