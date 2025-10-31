#!/usr/bin/env python3
"""
Property-based test demonstrating the IndexError bug in
django.templatetags.i18n.GetLanguageInfoListNode.get_language_info
"""

import sys

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings
from django.conf import settings
settings.configure(
    INSTALLED_APPS=['django.contrib.contenttypes'],
    LANGUAGES=[('en', 'English'), ('es', 'Spanish')],
    USE_I18N=True,
    USE_TZ=False,
    SECRET_KEY='test-secret-key'
)

# Initialize Django
import django
django.setup()

from hypothesis import given, strategies as st
from django.templatetags.i18n import GetLanguageInfoListNode

@given(st.one_of(
    st.just(''),
    st.just(()),
    st.just([]),
))
def test_get_language_info_should_not_crash_on_empty_input(language):
    """Test that get_language_info handles empty inputs gracefully"""
    node = GetLanguageInfoListNode(None, None)

    try:
        result = node.get_language_info(language)
        # If we get here without crashing, that's okay
    except IndexError:
        # This is the bug - we're catching IndexError and re-raising it
        # to make the test fail and demonstrate the bug
        raise

# Run the test
if __name__ == "__main__":
    test_get_language_info_should_not_crash_on_empty_input()