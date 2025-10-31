import numpy as np
from pandas.arrays import SparseArray

test_cases = [
    ([0, 1, 2, 2], 2),  # Original failing case
    ([1], 1),  # Hypothesis minimal case
    ([0, 0, 0, 0], 0),  # All zeros, fill=0 (should work)
    ([1, 1, 1, 1], 0),  # All nonzero, fill=0 (should work)
    ([1, 1, 1, 1], 1),  # All same as fill
    ([3, 3, 3, 3], 3),  # All same as nonzero fill
    ([-1, 0, 1, -1], -1),  # Negative nonzero fill
    ([0, 5, 0, 5], 5),  # Mixed with nonzero fill
]

for data, fill in test_cases:
    arr = SparseArray(data, fill_value=fill)
    sparse_result = arr.nonzero()[0]
    dense_result = arr.to_dense().nonzero()[0]

    match = np.array_equal(sparse_result, dense_result)
    status = "✓" if match else "✗"

    print(f"{status} Data: {data}, Fill: {fill}")
    print(f"  Dense nonzero indices: {list(dense_result)}")
    print(f"  Sparse nonzero indices: {list(sparse_result)}")
    if not match:
        print(f"  MISMATCH: Missing {set(dense_result) - set(sparse_result)}")
    print()