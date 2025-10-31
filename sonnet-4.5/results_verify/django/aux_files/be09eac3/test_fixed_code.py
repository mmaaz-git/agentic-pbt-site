import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
        USE_I18N=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French')],
    )
    django.setup()

from django.utils import translation

class MockGetLanguageInfoListNode:
    def __init__(self):
        pass

    def get_language_info_original(self, language):
        # Original buggy code
        if len(language[0]) > 1:
            return translation.get_language_info(language[0])
        else:
            return translation.get_language_info(str(language))

    def get_language_info_fixed(self, language):
        # Fixed code from the bug report
        if language and len(language[0]) > 1:
            return translation.get_language_info(language[0])
        else:
            return translation.get_language_info(str(language))

node = MockGetLanguageInfoListNode()

print("Testing ORIGINAL code with empty string:")
try:
    result = node.get_language_info_original('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting FIXED code with empty string:")
try:
    result = node.get_language_info_fixed('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting ORIGINAL code with empty list:")
try:
    result = node.get_language_info_original([])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting FIXED code with empty list:")
try:
    result = node.get_language_info_fixed([])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting ORIGINAL code with valid language code 'en':")
try:
    result = node.get_language_info_original('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting FIXED code with valid language code 'en':")
try:
    result = node.get_language_info_fixed('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")