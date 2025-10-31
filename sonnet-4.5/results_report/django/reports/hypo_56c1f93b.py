import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French'), ('de', 'German')],
        USE_I18N=True,
        USE_L10N=True,
        SECRET_KEY='test-secret-key',
        INSTALLED_APPS=[],
    )
    django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode


@given(st.one_of(
    st.text(min_size=0, max_size=10),
    st.lists(st.text(), min_size=0, max_size=5),
    st.tuples(st.text(min_size=0, max_size=5), st.text()),
))
def test_get_language_info_with_arbitrary_input(language):
    node = GetLanguageInfoListNode(None, 'result')
    result = node.get_language_info(language)
    assert result is not None

# Run the test
if __name__ == "__main__":
    test_get_language_info_with_arbitrary_input()