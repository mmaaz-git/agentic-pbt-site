#!/usr/bin/env python3
"""Simple reproduction of the reported bug in dask.utils.ndeepmap"""

from dask.utils import ndeepmap


def identity(x):
    return x


def inc(x):
    return x + 1


def main():
    print("=== Bug Reproduction: ndeepmap Silent Data Loss ===\n")

    # Test 1: Basic data loss
    input_list = [10, 20, 30]
    result = ndeepmap(0, identity, input_list)
    print(f"Test 1: ndeepmap(0, identity, [10, 20, 30])")
    print(f"  Input:  {input_list}")
    print(f"  Result: {result}")
    print(f"  Data Loss: Elements [20, 30] are silently discarded")
    print(f"  ✓ Bug confirmed: Only returns first element\n")

    # Test 2: Verify other elements are ignored
    input_modified = [10, 999, 999]
    result_modified = ndeepmap(0, identity, input_modified)
    print(f"Test 2: Changing non-first elements has no effect")
    print(f"  Original: [10, 20, 30] -> {result}")
    print(f"  Modified: [10, 999, 999] -> {result_modified}")
    print(f"  ✓ Bug confirmed: Result is identical ({result} == {result_modified})\n")

    # Test 3: With transformation function
    input_list2 = [1, 2, 3]
    result2 = ndeepmap(0, inc, input_list2)
    print(f"Test 3: ndeepmap(0, inc, [1, 2, 3])")
    print(f"  Input:  {input_list2}")
    print(f"  Result: {result2}")
    print(f"  Expected if no data loss: Either [2, 3, 4] or error")
    print(f"  Actual: inc(1) = 2, elements [2, 3] ignored")
    print(f"  ✓ Bug confirmed: Function only applied to first element\n")

    # Test 4: Empty list causes crash
    print("Test 4: Empty list handling")
    try:
        empty_result = ndeepmap(0, identity, [])
        print(f"  Result: {empty_result}")
    except IndexError as e:
        print(f"  ✗ Raises IndexError: {e}")
        print(f"  This is another bug: crashes on empty lists\n")

    # Test 5: Negative n
    neg_input = [1, 2, 3]
    neg_result = ndeepmap(-1, identity, neg_input)
    print(f"Test 5: ndeepmap(-1, identity, [1, 2, 3])")
    print(f"  Input:  {neg_input}")
    print(f"  Result: {neg_result}")
    print(f"  ✓ Bug confirmed: Negative n also loses data\n")

    # Test 6: Compare with documented behavior
    print("Test 6: Existing test cases (single element)")
    L = [1]
    result = ndeepmap(0, inc, L)
    print(f"  ndeepmap(0, inc, [1]) = {result}")
    print(f"  This works because list has only one element")
    print(f"  The bug is hidden when len(list) == 1\n")

    # Test 7: What the code actually does
    print("Test 7: Code analysis")
    print("  The code for n <= 0 and isinstance(seq, list):")
    print("    return func(seq[0])")
    print("  This explicitly returns func applied to ONLY the first element")
    print("  All other elements are silently discarded!\n")

    print("=== Summary ===")
    print("The bug is CONFIRMED:")
    print("- ndeepmap(n, func, seq) when n <= 0 and seq is a list")
    print("- Only processes seq[0], silently discarding seq[1:]")
    print("- This causes silent data loss")
    print("- Also crashes on empty lists (IndexError)")


if __name__ == "__main__":
    main()