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

django_settings.SECURE_REFERRER_POLICY = [" no-referrer ", "  strict-origin"]
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"List input: {django_settings.SECURE_REFERRER_POLICY}")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: ' no-referrer ,  strict-origin' (whitespace preserved)\n")

django_settings.SECURE_REFERRER_POLICY = " no-referrer ,  strict-origin"
middleware = SecurityMiddleware(Mock())
request = Mock(spec=HttpRequest)
request.is_secure.return_value = False
response = HttpResponse()
result = middleware.process_response(request, response)

print(f"String input: '{django_settings.SECURE_REFERRER_POLICY}'")
print(f"Header value: '{result.get('Referrer-Policy')}'")
print(f"Result: 'no-referrer,strict-origin' (whitespace stripped)")