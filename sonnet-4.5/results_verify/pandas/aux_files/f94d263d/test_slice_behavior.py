from pandas.core.indexers import length_of_indexer
import numpy as np

# Test how slices are handled (line 316 uses correct formula)
print("Testing slice handling vs range handling:\n")

test_data = np.arange(100)

cases = [
    (0, 1, 2),
    (0, 5, 3),
    (0, 10, 7),
    (1, 0, 1),  # Empty
    (0, 100, 7),
]

print("Comparison of slice vs range:")
for start, stop, step in cases:
    # Test with slice
    s = slice(start, stop, step)
    slice_result = length_of_indexer(s, target=test_data)

    # Test with range
    r = range(start, stop, step)
    range_result = length_of_indexer(r)

    # Python's actual behavior
    actual_range_len = len(list(r))
    actual_slice_len = len(test_data[s])

    print(f"  slice({start}, {stop}, {step}): pandas={slice_result}, actual={actual_slice_len}")
    print(f"  range({start}, {stop}, {step}): pandas={range_result}, actual={actual_range_len}")
    print(f"  Match: slice={'✓' if slice_result == actual_slice_len else '✗'}, "
          f"range={'✓' if range_result == actual_range_len else '✗'}\n")