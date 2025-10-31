from hypothesis import given, strategies as st
from django.utils.translation import to_locale


@given(st.text(min_size=4, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
def test_to_locale_without_dash_should_be_lowercase(language_str):
    result = to_locale(language_str)
    assert result == result.lower(), f"to_locale({language_str!r}) = {result!r}, but should be all lowercase when no dash present"


if __name__ == "__main__":
    test_to_locale_without_dash_should_be_lowercase()