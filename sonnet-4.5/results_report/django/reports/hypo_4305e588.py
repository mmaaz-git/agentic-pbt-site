import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        USE_I18N=True,
        LANGUAGES=[],
        LANGUAGE_CODE='en',
    )

import django
django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(
    lang_code=st.text(min_size=1, max_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    lang_name=st.text(min_size=1, max_size=20)
)
def test_single_char_tuple(lang_code, lang_name):
    node = GetLanguageInfoListNode(None, 'test_var')
    language = (lang_code, lang_name)

    # The function should extract the first element of the tuple
    # But due to the bug, it will pass str(language) instead
    try:
        result = node.get_language_info(language)
        # If it succeeds, check that the code matches
        assert result.get('code') == lang_code, f"Expected code {lang_code}, got {result.get('code')}"
    except KeyError as e:
        # The error message will contain the string representation of the tuple
        # instead of just the language code
        error_msg = str(e)
        if str(language) in error_msg:
            print(f"BUG CONFIRMED: Error message contains tuple string representation")
            print(f"  Input tuple: {language}")
            print(f"  Error message: {error_msg}")
            print(f"  Expected to look up: '{lang_code}'")
            print(f"  Actually looked up: '{str(language)}'")
            raise AssertionError(f"Bug detected: function passed '{str(language)}' instead of '{lang_code}'")
        raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test to find the bug...")
    print("=" * 60)
    try:
        test_single_char_tuple()
        print("No failures found after running multiple test cases")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("\nThis demonstrates the bug in GetLanguageInfoListNode.get_language_info()")
        print("The method incorrectly handles tuples with single-character language codes.")