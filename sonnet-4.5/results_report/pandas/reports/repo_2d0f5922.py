from pandas.api.indexers import FixedForwardWindowIndexer

# Create an indexer with a window size of 5
indexer = FixedForwardWindowIndexer(window_size=5)

# Try to get window bounds with step=0 (this should crash)
try:
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print(f"Success: start={start}, end={end}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()