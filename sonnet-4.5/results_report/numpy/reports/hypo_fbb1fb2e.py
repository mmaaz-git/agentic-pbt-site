from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.strings as nps

text_no_null = st.text(alphabet=st.characters(blacklist_characters='\x00'), min_size=1, max_size=20)

@given(text_no_null, text_no_null.filter(lambda x: len(x) > 0))
@settings(max_examples=500)
def test_partition_matches_python(s, sep):
    arr = np.array([s])
    sep_arr = np.array([sep])

    part1, part2, part3 = nps.partition(arr, sep_arr)
    p1, p2, p3 = s.partition(sep)

    assert part1[0] == p1 and part2[0] == p2 and part3[0] == p3

if __name__ == "__main__":
    test_partition_matches_python()