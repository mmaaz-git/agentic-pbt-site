import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import re
from unittest.mock import patch
from hypothesis import given, strategies as st, settings
import django
from django.conf import settings as django_settings

# Configure Django settings
django_settings.configure(DEBUG=True)

from django.conf.urls.static import static


@given(st.text(alphabet='/', min_size=1, max_size=10))
@settings(max_examples=100)
def test_all_slash_prefix_should_not_create_catchall(prefix):
    """
    Property: A prefix consisting only of slashes should either:
    1. Raise an error (like empty string does), OR
    2. Create a pattern that doesn't match unrelated URLs
    """
    with patch('django.conf.urls.static.settings') as mock_settings:
        mock_settings.DEBUG = True
        result = static(prefix)

        if result:
            pattern_regex = result[0].pattern._regex
            stripped = prefix.lstrip('/')
            if stripped == "":
                assert False, (
                    f"BUG: prefix {repr(prefix)} creates catch-all pattern. "
                    f"After lstrip('/'), prefix becomes empty, creating '^(?P<path>.*)$'"
                )

# Run the test
test_all_slash_prefix_should_not_create_catchall()