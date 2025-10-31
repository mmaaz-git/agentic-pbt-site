from pandas.api.indexers import FixedForwardWindowIndexer

# Test 1: Simple reproduction from bug report
print("Test 1: Reproducing the bug with step=0")
try:
    indexer = FixedForwardWindowIndexer(window_size=5)
    start, end = indexer.get_window_bounds(num_values=10, step=0)
    print("No error raised - unexpected!")
except ZeroDivisionError as e:
    print(f"Got ZeroDivisionError: {e}")
except ValueError as e:
    print(f"Got ValueError: {e}")
except Exception as e:
    print(f"Got unexpected exception: {type(e).__name__}: {e}")

# Test 2: With the specific inputs from the property-based test
print("\nTest 2: Testing with window_size=1, num_values=1, step=0")
try:
    indexer = FixedForwardWindowIndexer(window_size=1)
    start, end = indexer.get_window_bounds(num_values=1, step=0)
    print("No error raised - unexpected!")
except ZeroDivisionError as e:
    print(f"Got ZeroDivisionError: {e}")
except ValueError as e:
    print(f"Got ValueError: {e}")
except Exception as e:
    print(f"Got unexpected exception: {type(e).__name__}: {e}")

# Test 3: Verify that step=1 works correctly
print("\nTest 3: Verifying normal operation with step=1")
try:
    indexer = FixedForwardWindowIndexer(window_size=5)
    start, end = indexer.get_window_bounds(num_values=10, step=1)
    print(f"Success! Got start shape: {start.shape}, end shape: {end.shape}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Test 4: Check what happens with negative step
print("\nTest 4: Testing with negative step=-1")
try:
    indexer = FixedForwardWindowIndexer(window_size=5)
    start, end = indexer.get_window_bounds(num_values=10, step=-1)
    print(f"Success! Got start shape: {start.shape}, end shape: {end.shape}")
except Exception as e:
    print(f"Got exception: {type(e).__name__}: {e}")