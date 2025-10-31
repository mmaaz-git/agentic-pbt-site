from hypothesis import given, strategies as st, settings, example
import numpy as np
import numpy.strings as nps

text_no_null = st.text(alphabet=st.characters(blacklist_characters='\x00'), min_size=1, max_size=20)

@given(text_no_null, text_no_null.filter(lambda x: len(x) > 0))
@settings(max_examples=500)
@example('0', '00')  # The failing example from the bug report
def test_rpartition_matches_python(s, sep):
    arr = np.array([s])
    sep_arr = np.array([sep])

    part1, part2, part3 = nps.rpartition(arr, sep_arr)
    p1, p2, p3 = s.rpartition(sep)

    assert part1[0] == p1 and part2[0] == p2 and part3[0] == p3, \
        f"NumPy result: ({part1[0]!r}, {part2[0]!r}, {part3[0]!r}) != Python result: ({p1!r}, {p2!r}, {p3!r})"

if __name__ == "__main__":
    test_rpartition_matches_python()
    print("All tests passed!")