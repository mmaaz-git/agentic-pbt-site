import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from unittest.mock import Mock

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        SECURE_REFERRER_POLICY=None,
        SECURE_HSTS_SECONDS=0,
        SECURE_HSTS_INCLUDE_SUBDOMAINS=False,
        SECURE_HSTS_PRELOAD=False,
        SECURE_CONTENT_TYPE_NOSNIFF=False,
        SECURE_SSL_REDIRECT=False,
        SECURE_SSL_HOST=None,
        SECURE_REDIRECT_EXEMPT=[],
        SECURE_CROSS_ORIGIN_OPENER_POLICY=None,
    )
    django.setup()

from django.middleware.security import SecurityMiddleware
from django.http import HttpResponse, HttpRequest

# Test case 1: Single space character
print("=" * 60)
print("Test Case 1: Single space character")
print("=" * 60)

# String input
django_settings.SECURE_REFERRER_POLICY = ' '
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)
header_from_str = result.get('Referrer-Policy')

print(f"String input: '{django_settings.SECURE_REFERRER_POLICY}'")
print(f"Header value: '{header_from_str}'")
print()

# List input
django_settings.SECURE_REFERRER_POLICY = [' ']
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)
header_from_list = result.get('Referrer-Policy')

print(f"List input: {django_settings.SECURE_REFERRER_POLICY}")
print(f"Header value: '{header_from_list}'")
print()

print(f"Headers match: {header_from_str == header_from_list}")
print()

# Test case 2: Multiple policies with whitespace
print("=" * 60)
print("Test Case 2: Multiple policies with whitespace")
print("=" * 60)

# List input with whitespace
django_settings.SECURE_REFERRER_POLICY = [" no-referrer ", "  strict-origin"]
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"List input: {django_settings.SECURE_REFERRER_POLICY}")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: Whitespace preserved in list values")
print()

# String input with whitespace
django_settings.SECURE_REFERRER_POLICY = " no-referrer ,  strict-origin"
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"String input: '{django_settings.SECURE_REFERRER_POLICY}'")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: Whitespace stripped from string values")