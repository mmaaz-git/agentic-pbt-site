from hypothesis import given, strategies as st
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1))
def test_looper_odd_even_properties(seq):
    result = list(looper(seq))
    for loop_obj, item in result:
        pos = loop_obj.index
        if pos % 2 == 0:
            assert loop_obj.even == True
            assert loop_obj.odd == False
        else:
            assert loop_obj.odd == True
            assert loop_obj.even == False


if __name__ == "__main__":
    test_looper_odd_even_properties()