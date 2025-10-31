"""Reproduce the IndexError bug in django.templatetags.static.PrefixNode.handle_token"""

import os
import sys
import django
from django.conf import settings

# Configure Django settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
            'django.contrib.staticfiles',
        ],
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        }],
        STATIC_URL='/static/',
        SECRET_KEY='test-secret-key',
    )
    django.setup()

from django.template import Template, Context, TemplateSyntaxError

print("Testing malformed template tag: {% get_static_prefix as %}")
print("-" * 60)

# Test 1: Reproduce the bug as stated in report
template_str = "{% load static %}{% get_static_prefix as %}"

try:
    template = Template(template_str)
    print("Template compiled successfully - This should NOT happen!")
except TemplateSyntaxError as e:
    print(f"Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"BUG CONFIRMED: IndexError raised instead of TemplateSyntaxError!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 60)
print("\nTesting valid template tags to ensure they still work:")
print("-" * 60)

# Test 2: Valid usage without 'as' clause
try:
    template = Template("{% load static %}{% get_static_prefix %}")
    result = template.render(Context())
    print(f"Valid tag without 'as': SUCCESS - Result: '{result}'")
except Exception as e:
    print(f"Valid tag without 'as': FAILED - {e}")

# Test 3: Valid usage with 'as' clause
try:
    template = Template("{% load static %}{% get_static_prefix as my_prefix %}{{ my_prefix }}")
    result = template.render(Context())
    print(f"Valid tag with 'as': SUCCESS - Result: '{result}'")
except Exception as e:
    print(f"Valid tag with 'as': FAILED - {e}")

# Test 4: Test with get_media_prefix (which uses the same code)
print("\n" + "-" * 60)
print("\nTesting get_media_prefix with same malformed pattern:")
print("-" * 60)

template_str = "{% load static %}{% get_media_prefix as %}"

try:
    template = Template(template_str)
    print("Template compiled successfully - This should NOT happen!")
except TemplateSyntaxError as e:
    print(f"Correctly raised TemplateSyntaxError: {e}")
except IndexError as e:
    print(f"BUG CONFIRMED: IndexError raised for get_media_prefix too!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("\nAnalysis of the bug:")
print("=" * 60)
print("In PrefixNode.handle_token (line 24-39 of static.py):")
print("- Line 30: tokens = token.contents.split()")
print("- When input is 'get_static_prefix as', tokens = ['get_static_prefix', 'as']")
print("- Line 31: checks if len(tokens) > 1 and tokens[1] != 'as' - passes (2 > 1 and 'as' == 'as')")
print("- Line 35: checks if len(tokens) > 1 - passes (2 > 1)")
print("- Line 36: varname = tokens[2] - INDEXERROR! (only tokens[0] and tokens[1] exist)")
print("\nThe condition on line 35 should be 'if len(tokens) > 2' or == 3")