#!/usr/bin/env python3

# Let's understand how Python handles slices with negative steps
test_list = [0, 1, 2, 3, 4]

print("Understanding Python's slice behavior with negative steps:")
print(f"Original list: {test_list}")
print()

# Test different slice configurations
slices = [
    slice(None, None, -1),
    slice(None, None, -2),
    slice(4, None, -1),
    slice(None, 0, -1),
    slice(4, 0, -1),
    slice(3, 1, -1),
]

for slc in slices:
    result = test_list[slc]
    # Use slice.indices to see how Python normalizes the slice
    start, stop, step = slc.indices(len(test_list))
    print(f"slice({slc.start}, {slc.stop}, {slc.step})")
    print(f"  Result: {result}")
    print(f"  Length: {len(result)}")
    print(f"  slice.indices(5) returns: start={start}, stop={stop}, step={step}")
    print(f"  len(range({start}, {stop}, {step})) = {len(range(start, stop, step))}")
    print()

# Now let's check what the pandas function is doing wrong
import pandas.core.indexers as indexers

print("\nComparing pandas length_of_indexer with correct calculation:")
for slc in slices:
    target = test_list
    computed = indexers.length_of_indexer(slc, target)
    actual = len(target[slc])

    # What Python slice.indices gives us
    start, stop, step = slc.indices(len(target))
    correct_length = len(range(start, stop, step))

    print(f"slice({slc.start}, {slc.stop}, {slc.step})")
    print(f"  pandas computed: {computed}")
    print(f"  actual length: {actual}")
    print(f"  using slice.indices: {correct_length}")
    print(f"  Match: pandas={computed == actual}, indices={correct_length == actual}")
    print()