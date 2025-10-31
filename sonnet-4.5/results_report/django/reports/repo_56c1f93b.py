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

from django.templatetags.i18n import GetLanguageInfoListNode

node = GetLanguageInfoListNode(None, 'result')

# Test with empty string
print("Testing with empty string '':")
try:
    result = node.get_language_info('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with empty list
print("\nTesting with empty list []:")
try:
    result = node.get_language_info([])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with empty tuple
print("\nTesting with empty tuple ():")
try:
    result = node.get_language_info(())
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# For comparison - test with valid language code
print("\nTesting with valid language code 'en':")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test with invalid but non-empty language code
print("\nTesting with invalid language code 'xyz':")
try:
    result = node.get_language_info('xyz')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")