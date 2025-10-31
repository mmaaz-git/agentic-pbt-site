#!/usr/bin/env python3
"""Test if the proposed fix works"""

import numpy as np
from dask.array.creation import normalize_chunks
from dask.base import tokenize
from dask.array import Array
from dask._task_spec import Task

def eye_fixed(N, chunks="auto", M=None, k=0, dtype=float):
    """Fixed version of eye function"""
    if M is None:
        M = N
    if dtype is None:
        dtype = float

    if not isinstance(chunks, (int, str)):
        raise ValueError("chunks must be an int or string")

    vchunks, hchunks = normalize_chunks(chunks, shape=(N, M), dtype=dtype)
    chunks = vchunks[0]

    token = tokenize(N, chunks, M, k, dtype)
    name_eye = "eye-" + token

    dsk = {}
    for i, vchunk in enumerate(vchunks):
        for j, hchunk in enumerate(hchunks):
            key = (name_eye, i, j)
            if (j - i - 1) * chunks <= k <= (j - i + 1) * chunks:
                t = Task(
                    key,
                    np.eye,
                    vchunk,
                    hchunk,
                    k - (j - i) * chunks,
                    dtype,
                )
            else:
                t = Task(key, np.zeros, (vchunk, hchunk), dtype)
            dsk[t.key] = t

    # THE FIX: Use (vchunks, hchunks) instead of (chunks, chunks)
    return Array(dsk, name_eye, shape=(N, M), chunks=(vchunks, hchunks), dtype=dtype)

print("Testing the fixed version:")
print("=" * 50)

# Test the failing case
print("\nTest case: N=2, M=3, chunks=3")
try:
    arr = eye_fixed(2, chunks=3, M=3, k=0)
    print(f"Created array with shape {arr.shape}")
    print(f"Chunks: {arr.chunks}")
    result = arr.compute()
    print(f"Computed successfully!")
    print(f"Result:\n{result}")
    print(f"Expected:\n{np.eye(2, M=3)}")
    print(f"Match: {np.allclose(result, np.eye(2, M=3))}")
except Exception as e:
    print(f"ERROR: {e}")

# Test other cases
test_cases = [
    (2, 3, 2, 0),
    (3, 3, 3, 0),
    (4, 2, 3, 0),
    (3, 4, 2, 1),  # with k=1
    (4, 3, 2, -1), # with k=-1
]

print("\nTesting additional cases:")
for N, M, chunks, k in test_cases:
    print(f"\nN={N}, M={M}, chunks={chunks}, k={k}")
    try:
        arr = eye_fixed(N, chunks=chunks, M=M, k=k)
        result = arr.compute()
        expected = np.eye(N, M=M, k=k)
        match = np.allclose(result, expected)
        print(f"  Shape: {arr.shape}, Chunks: {arr.chunks}")
        print(f"  Match with numpy: {match}")
        if not match:
            print(f"  Result:\n{result}")
            print(f"  Expected:\n{expected}")
    except Exception as e:
        print(f"  ERROR: {e}")