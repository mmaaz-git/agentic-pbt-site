from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import chars_to_ranges

@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_chars_to_ranges_coverage(s):
    ranges = chars_to_ranges(s)

    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i+1]
        for code in range(code1, code2):
            covered_chars.add(chr(code))

    assert set(s) == covered_chars

if __name__ == "__main__":
    test_chars_to_ranges_coverage()