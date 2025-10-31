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