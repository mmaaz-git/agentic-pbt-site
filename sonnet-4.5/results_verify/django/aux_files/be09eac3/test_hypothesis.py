from hypothesis import given, strategies as st, example
from django.templatetags.i18n import GetLanguageInfoListNode
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French'), ('de', 'German')],
    )

@given(st.one_of(
    st.text(min_size=0, max_size=10),
    st.lists(st.text(), min_size=0, max_size=5),
    st.tuples(st.text(min_size=0, max_size=5), st.text()),
))
@example('')  # Force test with empty string
def test_get_language_info_with_arbitrary_input(language):
    node = GetLanguageInfoListNode(None, 'result')
    try:
        result = node.get_language_info(language)
        assert result is not None
        print(f"Success with input: {repr(language)}")
    except Exception as e:
        print(f"Failed with input: {repr(language)}")
        print(f"  Exception: {type(e).__name__}: {e}")
        raise

# Run the test
if __name__ == "__main__":
    test_get_language_info_with_arbitrary_input()