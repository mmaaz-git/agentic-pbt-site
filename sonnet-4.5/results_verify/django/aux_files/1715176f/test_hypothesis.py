"""Run the hypothesis-based property test from the bug report"""

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

from hypothesis import given, strategies as st, assume
from django.template import Template, Context, TemplateSyntaxError

@given(st.text(min_size=1, max_size=50))
def test_prefix_node_malformed_as_clause(text):
    assume('"' not in text and "'" not in text)
    assume(' ' not in text and text != 'as')

    template_str = "{% load static %}{% get_static_prefix as %}"

    try:
        template = Template(template_str)
        # If it compiles, it should work without error
        result = template.render(Context())
    except TemplateSyntaxError:
        # This is expected for malformed templates
        pass
    except IndexError:
        # This should never happen - it's the bug
        assert False, "Should raise TemplateSyntaxError, not IndexError"

# Run the test
print("Running hypothesis test...")
test_prefix_node_malformed_as_clause()
print("Test completed successfully if you see this message!")