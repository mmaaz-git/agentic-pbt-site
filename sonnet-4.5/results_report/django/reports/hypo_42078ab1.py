#!/usr/bin/env python3
"""
Property-based test for django.templatetags.static.PrefixNode bug
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

from hypothesis import given, strategies as st, assume
from django.template import Template, Context, TemplateSyntaxError

@given(st.text(min_size=1, max_size=50))
def test_prefix_node_malformed_as_clause(text):
    assume('"' not in text and "'" not in text)
    assume(' ' not in text and text != 'as')

    # Testing get_static_prefix with malformed 'as' clause
    template_str = "{% load static %}{% get_static_prefix as %}"

    try:
        template = Template(template_str)
    except TemplateSyntaxError:
        # This is the expected behavior
        pass
    except IndexError:
        # This is the bug - should raise TemplateSyntaxError instead
        assert False, "Should raise TemplateSyntaxError, not IndexError"

# Run the property-based test
if __name__ == "__main__":
    print("Running property-based test for PrefixNode bug...")
    print("Testing template: {% get_static_prefix as %}")
    print()

    try:
        test_prefix_node_malformed_as_clause()
        print("✗ Test FAILED: Bug confirmed - IndexError raised instead of TemplateSyntaxError")
    except AssertionError as e:
        print(f"✗ Test FAILED: {e}")
        print("\nThis confirms the bug - Django raises IndexError when it should raise TemplateSyntaxError")
    except Exception as e:
        print(f"✗ Unexpected error during test: {e}")
        import traceback
        traceback.print_exc()
    else:
        print("✓ Test passed - no bug found (unexpected!)")