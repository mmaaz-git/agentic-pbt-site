"""Test to reproduce the reported CSRF scheme validation bug"""
import django
from django.conf import settings

# Configure Django
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test',
        CSRF_TRUSTED_ORIGINS=[]
    )
    django.setup()

from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins
from unittest.mock import patch
from urllib.parse import urlsplit

print("=" * 60)
print("Testing CSRF scheme validation bug")
print("=" * 60)

# Test various malformed origins
test_cases = [
    "a://a",  # Invalid scheme
    "example.com://http",  # Scheme in wrong position
    "://example.com",  # Empty scheme
    "test://test",  # Unknown scheme
    "ftp://example.com",  # Valid URL structure but potentially unsupported scheme
    "http://example.com",  # Valid HTTP
    "https://example.com",  # Valid HTTPS
    "ws://example.com",  # WebSocket scheme
    "wss://example.com",  # Secure WebSocket scheme
]

for origin in test_cases:
    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        errors = check_csrf_trusted_origins(app_configs=None)
        parsed = urlsplit(origin)

        print(f"\nOrigin: '{origin}'")
        print(f"  Errors detected: {len(errors)}")
        print(f"  urlsplit results:")
        print(f"    - scheme: '{parsed.scheme}'")
        print(f"    - netloc: '{parsed.netloc}'")
        print(f"    - path: '{parsed.path}'")

        if errors:
            for error in errors:
                print(f"  Error message: {error.msg}")

print("\n" + "=" * 60)
print("Testing how middleware would use these origins")
print("=" * 60)

# Test how the middleware code would handle these
from django.middleware.csrf import CsrfViewMiddleware

middleware = CsrfViewMiddleware(lambda x: None)

for origin in ["a://a", "http://example.com", "example.com://http"]:
    try:
        with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
            # Clear cached properties
            if hasattr(middleware, '_csrf_trusted_origins_hosts'):
                delattr(middleware, '_csrf_trusted_origins_hosts')
            if hasattr(middleware, '_allowed_origins_exact'):
                delattr(middleware, '_allowed_origins_exact')

            parsed = urlsplit(origin)
            print(f"\nOrigin: '{origin}'")
            print(f"  urlsplit scheme: '{parsed.scheme}', netloc: '{parsed.netloc}'")

            # Try to access the property that would be used
            hosts = middleware.csrf_trusted_origins_hosts
            print(f"  Extracted host would be: '{hosts[0] if hosts else 'None'}'")

    except Exception as e:
        print(f"  ERROR when processing: {e}")