import numpy as np
import numpy.strings as ns
from hypothesis import given, settings, strategies as st
import sys

# Create a strategy for generating string arrays with a substring from them
string_arrays_with_substring = st.lists(st.text(), min_size=1, max_size=10).flatmap(
    lambda strings: st.tuples(st.just(np.array(strings)), st.sampled_from(strings))
)

failures = []

@given(string_arrays_with_substring)
@settings(max_examples=100)
def test_find_index_consistency(arr_and_sub):
    arr, sub = arr_and_sub
    find_result = ns.find(arr, sub)

    for i in range(len(arr)):
        if find_result[i] >= 0:
            try:
                index_result = ns.index(arr, sub)
                # If index succeeded, check that results match for this element
                if find_result[i] != index_result[i]:
                    failures.append((arr, sub, find_result, index_result))
                    assert False, f"find returned {find_result[i]} but index returned {index_result[i]} for element {i}"
            except ValueError:
                # Index raised ValueError even though find found the substring in at least one element
                failures.append((arr, sub, find_result, "ValueError"))
                assert False, f"find returned {find_result[i]} for element {i} but index raised ValueError"

try:
    test_find_index_consistency()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
    if failures:
        print(f"\nFirst failure case:")
        arr, sub, find_res, index_res = failures[0]
        print(f"  Array: {repr(arr)}")
        print(f"  Substring: {repr(sub)}")
        print(f"  find result: {find_res}")
        print(f"  index result: {index_res}")
except Exception as e:
    print(f"Test error: {e}")
    sys.exit(1)