import os
import sys

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from hypothesis import given, strategies as st
from django.conf.urls.static import static
import re


@given(st.sampled_from(["/", "//", "///", "////"]))
def test_slash_only_prefix_creates_overly_broad_pattern(prefix):
    """
    Property: Prefixes that become empty after lstrip("/") should be rejected,
    just like empty strings are rejected.
    """
    result = static(prefix)

    if result:
        pattern = result[0].pattern.regex

        lstripped = prefix.lstrip("/")
        assert lstripped == ""

        assert pattern.pattern == r'^(?P<path>.*)$'

        assert pattern.match("admin/")
        assert pattern.match("api/users/123")
        assert pattern.match("any/arbitrary/url")


# Run the test
if __name__ == "__main__":
    test_slash_only_prefix_creates_overly_broad_pattern()