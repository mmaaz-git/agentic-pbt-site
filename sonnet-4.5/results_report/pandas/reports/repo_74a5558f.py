from pandas.api.indexers import FixedForwardWindowIndexer

# Create a FixedForwardWindowIndexer with window_size=5
indexer = FixedForwardWindowIndexer(window_size=5)

# Attempt to get window bounds with step=0
# This should raise a descriptive ValueError but instead raises ZeroDivisionError
try:
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print("No error raised - this is unexpected!")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Unexpected error type {type(e).__name__}: {e}")