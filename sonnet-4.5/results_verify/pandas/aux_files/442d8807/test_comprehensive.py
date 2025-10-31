import numpy as np
from pandas.core.indexers import length_of_indexer

def test_slice_behavior():
    """Test how Python's native slice behaves with different indices"""

    print("=== Testing Python's native slice behavior ===\n")

    test_cases = [
        (np.array([0]), slice(2, None, None), "Out-of-bounds start"),
        (np.array([0]), slice(None, -2, None), "Negative stop before start"),
        (np.array([1, 2, 3]), slice(5, 10, None), "Both start and stop out of bounds"),
        (np.array([1, 2, 3, 4, 5]), slice(3, 2, None), "Start > stop"),
        (np.array([1, 2, 3]), slice(0, 0, None), "Start == stop"),
        (np.array([1, 2, 3]), slice(10, None, None), "Start way out of bounds"),
        (np.array([1, 2, 3]), slice(None, -10, None), "Negative stop way out of bounds"),
    ]

    for array, indexer, description in test_cases:
        actual_result = array[indexer]
        actual_length = len(actual_result)
        predicted_length = length_of_indexer(indexer, array)

        print(f"{description}:")
        print(f"  Array: {array}, Length: {len(array)}")
        print(f"  Indexer: {indexer}")
        print(f"  Actual result: {actual_result}")
        print(f"  Actual length: {actual_length}")
        print(f"  Predicted length: {predicted_length}")
        print(f"  Correct: {actual_length == predicted_length}")
        print(f"  Negative length bug: {predicted_length < 0}\n")

def test_conceptual_issue():
    """Test to understand the conceptual issue"""
    print("\n=== Understanding the conceptual issue ===\n")
    print("The length of a slice result is always >= 0 by definition.")
    print("Python's len() function returns the number of elements, which cannot be negative.")
    print("Therefore, length_of_indexer should NEVER return a negative value.\n")

    # Show that Python never creates negative-length slices
    arrays = [
        np.array([1]),
        np.array([1, 2, 3]),
        np.array(range(10))
    ]

    slices = [
        slice(100, None),
        slice(None, -100),
        slice(50, 10),
        slice(-50, -100),
    ]

    print("Testing various edge cases with Python slicing:")
    for arr in arrays:
        for s in slices:
            result = arr[s]
            print(f"  array[{len(arr)}][{s}] -> len={len(result)} (never negative!)")

if __name__ == "__main__":
    test_slice_behavior()
    test_conceptual_issue()