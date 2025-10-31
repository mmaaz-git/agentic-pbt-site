from hypothesis import given, strategies as st
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1))
def test_looper_odd_even_properties(seq):
    result = list(looper(seq))
    for loop_obj, item in result:
        pos = loop_obj.index
        if pos % 2 == 0:
            assert loop_obj.even == True, f"Position {pos}: even should be True but is {loop_obj.even}"
            assert loop_obj.odd == False, f"Position {pos}: odd should be False but is {loop_obj.odd}"
        else:
            assert loop_obj.odd == True, f"Position {pos}: odd should be True but is {loop_obj.odd}"
            assert loop_obj.even == False, f"Position {pos}: even should be False but is {loop_obj.even}"

# Test with the specific failing input
def test_single_case(seq):
    result = list(looper(seq))
    for loop_obj, item in result:
        pos = loop_obj.index
        if pos % 2 == 0:
            assert loop_obj.even == True, f"Position {pos}: even should be True but is {loop_obj.even}"
            assert loop_obj.odd == False, f"Position {pos}: odd should be False but is {loop_obj.odd}"
        else:
            assert loop_obj.odd == True, f"Position {pos}: odd should be True but is {loop_obj.odd}"
            assert loop_obj.even == False, f"Position {pos}: even should be False but is {loop_obj.even}"

print("Testing with seq=[0]:")
try:
    test_single_case([0])
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

print("\nTesting with seq=[1, 2, 3]:")
try:
    test_single_case([1, 2, 3])
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")