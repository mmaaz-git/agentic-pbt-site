import os
import sys

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static

# Test with "/" prefix
print("Testing static('/') function:")
print("-" * 40)

result = static("/")
if result:
    pattern = result[0].pattern.regex
    print(f"Generated pattern: {pattern.pattern}")
    print()

    # Test various URLs against the pattern
    test_urls = [
        "",
        "admin/",
        "api/users/123",
        "any/arbitrary/url",
        "static/file.css",
        "media/image.png"
    ]

    print("Pattern matching results:")
    for url in test_urls:
        match = pattern.match(url)
        print(f"  '{url}': {bool(match)}")

    print()
    print("✗ BUG CONFIRMED: The pattern '^(?P<path>.*)$' matches ALL URLs!")
    print("This would cause the static file handler to intercept every request in the Django application.")
else:
    print("No pattern generated (this shouldn't happen in DEBUG mode)")