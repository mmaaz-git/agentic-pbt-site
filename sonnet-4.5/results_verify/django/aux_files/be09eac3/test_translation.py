from django.utils import translation
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French')],
    )

print("Testing translation.get_language_info with empty string:")
try:
    result = translation.get_language_info('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting translation.get_language_info with 'en':")
try:
    result = translation.get_language_info('en')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")