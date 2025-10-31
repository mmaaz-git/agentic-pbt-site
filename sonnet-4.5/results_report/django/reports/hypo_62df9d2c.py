from hypothesis import given, strategies as st, assume
from django.utils.datastructures import CaseInsensitiveMapping

@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_caseinsensitivemapping_case_insensitive(d):
    assume(len(d) > 0)
    mapping = CaseInsensitiveMapping(d)

    for key in d:
        assert mapping[key] == d[key]
        assert mapping[key.upper()] == d[key]
        assert mapping[key.lower()] == d[key]

if __name__ == "__main__":
    test_caseinsensitivemapping_case_insensitive()