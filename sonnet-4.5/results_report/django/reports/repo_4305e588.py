import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        USE_I18N=True,
        LANGUAGES=[('x', 'Language X')],  # Custom single-character language code
        LANGUAGE_CODE='en',
    )

import django
django.setup()

from django.templatetags.i18n import GetLanguageInfoListNode

# Create an instance of GetLanguageInfoListNode
node = GetLanguageInfoListNode(None, 'test_var')

# Test case 1: Works correctly with two-character code
print("Test 1: Two-character language code in tuple")
print("Input: ('en', 'English')")
try:
    result = node.get_language_info(('en', 'English'))
    print(f"Result: {result}")
    print(f"Success - extracted language code correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 2: Fails with single-character code
print("Test 2: Single-character language code in tuple")
print("Input: ('x', 'Language X')")
try:
    result = node.get_language_info(('x', 'Language X'))
    print(f"Result: {result}")
    print(f"Success - extracted language code correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 3: String input with two-character code
print("Test 3: String input with two-character code")
print("Input: 'en'")
try:
    result = node.get_language_info('en')
    print(f"Result: {result}")
    print(f"Success - processed string correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Test case 4: String input with single-character code (hypothetical)
print("Test 4: String input with single-character (hypothetical)")
print("Input: 'x'")
try:
    result = node.get_language_info('x')
    print(f"Result: {result}")
    print(f"Success - processed string correctly\n")
except Exception as e:
    print(f"Error: {e}\n")

# Demonstrate the bug: what actually gets passed to get_language_info
print("Debug: What gets passed to translation.get_language_info()")
print("------------------------------------------------------")
test_tuple = ('x', 'Language X')
print(f"For tuple {test_tuple}:")
print(f"  language[0] = '{test_tuple[0]}' (length = {len(test_tuple[0])})")
print(f"  len(language[0]) > 1 evaluates to: {len(test_tuple[0]) > 1}")
if len(test_tuple[0]) > 1:
    print(f"  Would pass: language[0] = '{test_tuple[0]}'")
else:
    print(f"  Would pass: str(language) = '{str(test_tuple)}'")
print(f"  The string '{str(test_tuple)}' is NOT a valid language code!")