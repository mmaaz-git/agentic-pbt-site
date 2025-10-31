#!/usr/bin/env python3
"""Analyze the chunking logic in dask.array.eye"""

import dask.array as da
from dask.array.creation import normalize_chunks

# Test normalize_chunks behavior
print("Testing normalize_chunks behavior:")
print("=" * 50)

test_cases = [
    (2, 3, 3),  # N=2, M=3, chunks=3 (failing case)
    (2, 3, 2),  # N=2, M=3, chunks=2 (working case)
    (3, 3, 3),  # N=3, M=3, chunks=3 (square, works)
    (4, 2, 3),  # N=4, M=2, chunks=3
]

for N, M, chunks in test_cases:
    print(f"\nN={N}, M={M}, chunks={chunks}")
    vchunks, hchunks = normalize_chunks(chunks, shape=(N, M), dtype=float)
    print(f"  vchunks (row chunks): {vchunks}")
    print(f"  hchunks (col chunks): {hchunks}")
    print(f"  vchunks[0]: {vchunks[0]}")
    print(f"  Using chunks=(vchunks[0], vchunks[0]) = ({vchunks[0]}, {vchunks[0]})")
    print(f"  Should be chunks=(vchunks, hchunks) = ({vchunks}, {hchunks})")

    # Check what tasks would be created
    print(f"  Tasks that would be created:")
    for i, vchunk in enumerate(vchunks):
        for j, hchunk in enumerate(hchunks):
            print(f"    Task ({i}, {j}) for chunk of size ({vchunk}, {hchunk})")

    # Check what the Array expects
    print(f"  Array constructor expects chunks=({vchunks[0]}, {vchunks[0]})")
    print(f"  This means it expects tasks at:")
    if vchunks[0] == vchunks[0]:  # Always true, but showing the logic
        # With chunks=(2, 2) for N=2, M=3
        if N == 2 and M == 3 and chunks == 3:
            print(f"    (0, 0) and (0, 1) based on shape (2, 3) with chunks (2, 2)")
            print(f"    But (0, 1) doesn't exist because we only created tasks based on actual chunks!")
            print(f"    The actual chunks are {hchunks} = (2, 1) for columns")
            print(f"    So task (0, 1) exists, but Array thinks chunks are (2, 2)")
            print(f"    This mismatch causes the 'Missing dependency' error")