import os
import django
from django.conf import settings as django_settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

if not django_settings.configured:
    django_settings.configure(
        DEBUG=False,
        SECRET_KEY='test-key-' + 'x' * 50,
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        INSTALLED_APPS=[],
        ALLOWED_HOSTS=['*'],
    )
    django.setup()

from django.core.checks.security.base import check_cross_origin_opener_policy

# Test without whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none"
result_valid = check_cross_origin_opener_policy(None)
print(f"Without whitespace: {result_valid}")

# Test with trailing whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = "unsafe-none "
result_with_whitespace = check_cross_origin_opener_policy(None)
print(f"With trailing whitespace: {result_with_whitespace}")

# Test with leading whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = " unsafe-none"
result_with_leading = check_cross_origin_opener_policy(None)
print(f"With leading whitespace: {result_with_leading}")

# Test with both leading and trailing whitespace
django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = " unsafe-none "
result_with_both = check_cross_origin_opener_policy(None)
print(f"With both whitespaces: {result_with_both}")