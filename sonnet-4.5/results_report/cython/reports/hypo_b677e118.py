from hypothesis import assume, given, strategies as st
from django.utils.translation import to_locale

@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=4, max_size=10))
def test_to_locale_case_invariance_no_dash(language):
    assume('-' not in language)

    result_upper = to_locale(language.upper())
    result_lower = to_locale(language.lower())

    assert result_upper == result_lower, f"to_locale should be case-insensitive"

if __name__ == "__main__":
    test_to_locale_case_invariance_no_dash()