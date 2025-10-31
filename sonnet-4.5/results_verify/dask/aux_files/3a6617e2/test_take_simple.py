#!/usr/bin/env python3
"""Simple test script to reproduce the dask.bag.Bag.take bug"""

import dask
import dask.bag as db
import warnings

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

print("=" * 60)
print("TEST 1: Reproducing the reported bug")
print("=" * 60)

items = [0, 0, 0]
bag = db.from_sequence(items, npartitions=2)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = bag.take(3, compute=True)

    print(f"Items: {items}")
    print(f"Total items in bag: {len(items)}")
    print(f"Number of partitions: 2")
    print(f"Requested: 3 elements")
    print(f"Got: {result}")
    print(f"Length: {len(result)}")

    if w:
        print(f"\nWarning message raised:")
        print(f"  {w[0].message}")

print("\n" + "=" * 60)
print("TEST 2: Same request with npartitions=2")
print("=" * 60)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result_2 = bag.take(3, npartitions=2, compute=True)

    print(f"Items: {items}")
    print(f"Total items in bag: {len(items)}")
    print(f"Requested: 3 elements with npartitions=2")
    print(f"Got: {result_2}")
    print(f"Length: {len(result_2)}")

    if w:
        print(f"Warning message: {w[0].message}")
    else:
        print("No warning raised - got all 3 elements correctly")

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)
print("The bag contains 3 elements total.")
print("With default npartitions=1, take() only searches the first partition.")
print("The first partition has 2 elements, so it returns 2.")
print("The warning says 'only 2 elements available' but that's misleading.")
print("There ARE 3 elements available total, just not in the first partition.")
print("\nThe warning message should clarify it only searched the first partition.")