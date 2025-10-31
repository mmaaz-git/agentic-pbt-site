#!/usr/bin/env python3
"""Test the reported bug with django.conf.urls.static"""

import django
from django.conf import settings as django_settings

# Configure Django if not already configured
if not django_settings.configured:
    django_settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured

print("Testing django.conf.urls.static.static() with various inputs...")
print("="*60)

# Test 1: Empty string (should raise)
print("\nTest 1: Empty string ''")
try:
    result = static('')
    print(f"  Result: {result}")
    print("  ERROR: Should have raised ImproperlyConfigured!")
except ImproperlyConfigured as e:
    print(f"  ✓ Raised ImproperlyConfigured: {e}")

# Test 2: Single space (bug report claims this doesn't raise but should)
print("\nTest 2: Single space ' '")
try:
    result = static(' ')
    print(f"  Result: {result}")
    print(f"  Type: {type(result)}")
    if result:
        print(f"  Pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised ImproperlyConfigured: {e}")

# Test 3: Tab character
print("\nTest 3: Tab character '\\t'")
try:
    result = static('\t')
    print(f"  Result: {result}")
    if result:
        print(f"  Pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised ImproperlyConfigured: {e}")

# Test 4: Newline character
print("\nTest 4: Newline character '\\n'")
try:
    result = static('\n')
    print(f"  Result: {result}")
    if result:
        print(f"  Pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised ImproperlyConfigured: {e}")

# Test 5: Multiple spaces
print("\nTest 5: Multiple spaces '   '")
try:
    result = static('   ')
    print(f"  Result: {result}")
    if result:
        print(f"  Pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised ImproperlyConfigured: {e}")

# Test 6: Valid prefix for comparison
print("\nTest 6: Valid prefix 'media/'")
try:
    result = static('media/')
    print(f"  Result: {result}")
    if result:
        print(f"  Pattern: {result[0].pattern if hasattr(result[0], 'pattern') else 'N/A'}")
except ImproperlyConfigured as e:
    print(f"  Raised ImproperlyConfigured: {e}")

print("\n" + "="*60)
print("Analysis complete")