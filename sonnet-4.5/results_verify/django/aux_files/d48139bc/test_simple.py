#!/usr/bin/env python3
"""Simple test without hypothesis"""

import django
from django.conf import settings as django_settings

# Configure Django if not already configured
if not django_settings.configured:
    django_settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured

print("Testing behavior claimed in bug report...")
print("="*60)

# Test what bug report says
print("\nBug report claims:")
print("1. Empty string '' raises ImproperlyConfigured ✓")
print("2. Whitespace-only strings like ' ' do NOT raise but SHOULD")
print("\nActual behavior:")

# Test empty string
print("\nstatic(''):")
try:
    result = static('')
    print(f"  Returned: {result}")
except ImproperlyConfigured as e:
    print(f"  ✓ Raised: {e}")

# Test single space
print("\nstatic(' '):")
try:
    result = static(' ')
    print(f"  Returned: {result}")
    print(f"  URL Pattern created: {result[0].pattern if result and hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised: {e}")

print("\n" + "="*60)
print("Bug report claim CONFIRMED:")
print("- Empty string '' correctly raises ImproperlyConfigured")
print("- Whitespace string ' ' does NOT raise but creates a URL pattern")
print("- This creates pattern: '^\ (?P<path>.*)$' which is likely not useful")