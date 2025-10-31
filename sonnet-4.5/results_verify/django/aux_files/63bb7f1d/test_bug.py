import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        STATIC_URL='/static/',
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
        }],
    )

import django
django.setup()

from django.templatetags.static import PrefixNode

class MockToken:
    def __init__(self, contents):
        self.contents = contents

class MockParser:
    pass

# Test case 1: The exact reproducer from bug report
print("Test 1: Reproducing with 'get_static_prefix as'")
try:
    token = MockToken('get_static_prefix as')
    parser = MockParser()
    result = PrefixNode.handle_token(parser, token, 'STATIC_URL')
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test case 2: The property-based test failure case
print("\nTest 2: Property-based test failure case '0 as'")
try:
    token = MockToken('0 as')
    parser = MockParser()
    result = PrefixNode.handle_token(parser, token, 'STATIC_URL')
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test case 3: Valid syntax for comparison
print("\nTest 3: Valid syntax 'get_static_prefix as myvar'")
try:
    token = MockToken('get_static_prefix as myvar')
    parser = MockParser()
    result = PrefixNode.handle_token(parser, token, 'STATIC_URL')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test case 4: No 'as' clause
print("\nTest 4: No 'as' clause 'get_static_prefix'")
try:
    token = MockToken('get_static_prefix')
    parser = MockParser()
    result = PrefixNode.handle_token(parser, token, 'STATIC_URL')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")