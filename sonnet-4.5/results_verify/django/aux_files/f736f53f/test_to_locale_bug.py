from hypothesis import given, strategies as st, settings, example
from django.utils.translation import to_locale


@given(st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', min_size=4, max_size=10))
@settings(max_examples=1000)
@example('GerMAN')
def test_to_locale_case_handling_without_dash(language):
    locale = to_locale(language)
    assert locale.islower(), (
        f"to_locale('{language}') = '{locale}' has inconsistent casing"
    )

if __name__ == "__main__":
    test_to_locale_case_handling_without_dash()
    print("Test completed")