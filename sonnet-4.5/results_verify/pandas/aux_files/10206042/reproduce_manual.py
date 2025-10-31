from pandas.api.indexers import FixedForwardWindowIndexer
import traceback

try:
    indexer = FixedForwardWindowIndexer(window_size=5)
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print(f"Result: start={start}, end={end}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()