from django.templatetags.i18n import GetLanguageInfoListNode
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        LANGUAGES=[('en', 'English'), ('fr', 'French')],
    )

node = GetLanguageInfoListNode(None, 'result')

print("Testing with empty string:")
try:
    result = node.get_language_info('')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting with empty list:")
try:
    result = node.get_language_info([])
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTesting with empty tuple:")
try:
    result = node.get_language_info(())
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")