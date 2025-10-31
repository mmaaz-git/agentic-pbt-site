#!/usr/bin/env python3
"""
Minimal test case demonstrating the IndexError bug in django.templatetags.static.PrefixNode
"""

import os
import sys
import django
from django.conf import settings

# Configure minimal Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': False,
            'OPTIONS': {
                'context_processors': [],
            },
        }],
        INSTALLED_APPS=[
            'django.contrib.staticfiles',
        ],
        STATIC_URL='/static/',
        MEDIA_URL='/media/',
    )
    django.setup()

from django.template import Template, Context, TemplateSyntaxError

# Test case 1: get_static_prefix with 'as' but no variable name
print("=" * 60)
print("Test 1: {% get_static_prefix as %}")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix as %}"

try:
    template = Template(template_str)
    print("✓ Template compiled successfully (unexpected!)")
except TemplateSyntaxError as e:
    print(f"✓ Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"✗ BUG: IndexError raised instead of TemplateSyntaxError")
    print(f"  Error message: {e}")
    import traceback
    print(f"  Traceback:")
    traceback.print_exc()

print()

# Test case 2: get_media_prefix with 'as' but no variable name
print("=" * 60)
print("Test 2: {% get_media_prefix as %}")
print("=" * 60)
template_str = "{% load static %}{% get_media_prefix as %}"

try:
    template = Template(template_str)
    print("✓ Template compiled successfully (unexpected!)")
except TemplateSyntaxError as e:
    print(f"✓ Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"✗ BUG: IndexError raised instead of TemplateSyntaxError")
    print(f"  Error message: {e}")
    import traceback
    print(f"  Traceback:")
    traceback.print_exc()

print()

# Test case 3: Valid usage without 'as' clause (should work)
print("=" * 60)
print("Test 3: {% get_static_prefix %} (valid usage)")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix %}"

try:
    template = Template(template_str)
    result = template.render(Context())
    print(f"✓ Template compiled and rendered successfully")
    print(f"  Result: '{result}'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print()

# Test case 4: Valid usage with 'as' clause (should work)
print("=" * 60)
print("Test 4: {% get_static_prefix as my_prefix %} (valid usage)")
print("=" * 60)
template_str = "{% load static %}{% get_static_prefix as my_prefix %}{{ my_prefix }}"

try:
    template = Template(template_str)
    result = template.render(Context())
    print(f"✓ Template compiled and rendered successfully")
    print(f"  Result: '{result}'")
except Exception as e:
    print(f"✗ Unexpected error: {e}")