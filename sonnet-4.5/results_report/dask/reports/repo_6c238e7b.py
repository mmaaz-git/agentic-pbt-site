#!/usr/bin/env python3
"""
Minimal reproduction case for dask.bytes.read_bytes duplicate offset bug.
This demonstrates how the function generates duplicate offsets when
not_zero=True and the file size is small relative to blocksize.
"""

# Simulate the exact logic from dask.bytes.core.read_bytes (lines 124-141)
def simulate_read_bytes_offsets(size, blocksize, not_zero):
    """
    Simulates the offset calculation logic from dask.bytes.core.read_bytes.
    This is extracted directly from the implementation.
    """
    # From lines 124-127: Adjust blocksize if needed
    if size % blocksize and size > blocksize:
        blocksize1 = size / (size // blocksize)
    else:
        blocksize1 = blocksize

    # From lines 128-130: Initialize
    place = 0
    off = [0]
    length = []

    # From lines 133-136: Generate offsets
    while size - place > (blocksize1 * 2) - 1:
        place += blocksize1
        off.append(int(place))
        length.append(off[-1] - off[-2])

    # From line 137: Add final length
    length.append(size - off[-1])

    # From lines 139-141: Apply not_zero adjustment
    if not_zero:
        off[0] = 1
        length[0] -= 1

    return off, length

# Reproduce the bug with the failing input
size = 2
blocksize = 1
not_zero = True

print("=== Reproducing dask.bytes.read_bytes duplicate offset bug ===")
print(f"Input parameters:")
print(f"  size = {size}")
print(f"  blocksize = {blocksize}")
print(f"  not_zero = {not_zero}")
print()

offsets, lengths = simulate_read_bytes_offsets(size, blocksize, not_zero)

print(f"Generated offsets: {offsets}")
print(f"Generated lengths: {lengths}")
print()

# Check for the bug: duplicate offsets
print("=== Bug Analysis ===")
if len(offsets) > 1 and offsets[0] == offsets[1]:
    print(f"BUG DETECTED: Duplicate offsets found!")
    print(f"  offsets[0] = {offsets[0]}")
    print(f"  offsets[1] = {offsets[1]}")
    print()
    print("This violates the invariant that offsets should be strictly increasing.")
    print("Two blocks would start reading from the same position in the file,")
    print("causing data duplication or loss.")
else:
    print("No duplicate offsets found.")

# Verify the invariant that offsets should be strictly increasing
print()
print("=== Offset Invariant Check ===")
for i in range(1, len(offsets)):
    if offsets[i] <= offsets[i-1]:
        print(f"INVARIANT VIOLATED at index {i}:")
        print(f"  offsets[{i-1}] = {offsets[i-1]}")
        print(f"  offsets[{i}] = {offsets[i]}")
        print(f"  Expected: offsets[{i}] > offsets[{i-1}]")
        break
else:
    print("All offsets are strictly increasing (invariant satisfied).")