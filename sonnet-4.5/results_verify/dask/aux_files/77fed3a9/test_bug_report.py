#!/usr/bin/env python3
"""Test the reported bug in dask.bytes.read_bytes"""

from hypothesis import given, strategies as st, settings, example
import traceback

# Test 1: Run the hypothesis test
print("=" * 60)
print("TEST 1: Running Hypothesis property-based test")
print("=" * 60)

@given(
    st.integers(min_value=1, max_value=100000),
    st.integers(min_value=1, max_value=10000),
    st.booleans()
)
@example(2, 1, True)  # The failing example from the bug report
@settings(max_examples=500)
def test_blocksize_calculation_invariants(size, blocksize, not_zero):
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
    length.append(size - off[-1])

    if not_zero:
        off[0] = 1
        length[0] -= 1

    for i in range(1, len(off)):
        assert off[i] > off[i-1], f"Offsets not increasing at index {i}: off={off}, size={size}, blocksize={blocksize}, not_zero={not_zero}"

try:
    test_blocksize_calculation_invariants()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed with assertion: {e}")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Running specific failing example")
print("=" * 60)

# Test 2: Run the specific failing example
size = 2
blocksize = 1
not_zero = True

print(f"Input: size={size}, blocksize={blocksize}, not_zero={not_zero}")

if size % blocksize and size > blocksize:
    blocksize1 = size / (size // blocksize)
else:
    blocksize1 = blocksize

place = 0
off = [0]
length = []

print(f"blocksize1 = {blocksize1}")

while size - place > (blocksize1 * 2) - 1:
    place += blocksize1
    off.append(int(place))
    length.append(off[-1] - off[-2])
    print(f"  Added offset: place={place}, off={off}")

length.append(size - off[-1])

print(f"Before not_zero adjustment: off = {off}, length = {length}")

if not_zero:
    off[0] = 1
    length[0] -= 1

print(f"After not_zero adjustment: off = {off}, length = {length}")
print(f"Bug demonstration: off[0] = {off[0]}, off[1] = {off[1]}")
print(f"Are offsets strictly increasing? {all(off[i] > off[i-1] for i in range(1, len(off)))}")

# Test 3: Try with different small values to see the pattern
print("\n" + "=" * 60)
print("TEST 3: Testing other small file sizes")
print("=" * 60)

test_cases = [
    (1, 1, True),
    (2, 1, True),
    (3, 1, True),
    (3, 2, True),
    (4, 2, True),
    (4, 3, True),
    (10, 5, True),
]

for size, blocksize, not_zero in test_cases:
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    place = 0
    off = [0]
    length = []

    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])
    length.append(size - off[-1])

    if not_zero:
        off[0] = 1
        length[0] -= 1

    is_valid = all(off[i] > off[i-1] for i in range(1, len(off)))
    status = "✓ VALID" if is_valid else "✗ BUG"
    print(f"size={size:2d}, blocksize={blocksize:2d}, not_zero={str(not_zero):5s} => off={off}, {status}")