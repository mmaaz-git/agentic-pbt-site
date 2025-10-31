import os
import sys

# Add Django to path
django_path = '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages'
sys.path.insert(0, django_path)

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
    print(f"Testing prefix: {repr(prefix)}")
    result = static(prefix)

    if result:
        pattern = result[0].pattern.regex

        lstripped = prefix.lstrip("/")
        assert lstripped == ""

        assert pattern.pattern == r'^(?P<path>.*)$'
        print(f"  Pattern: {pattern.pattern}")

        assert pattern.match("admin/")
        assert pattern.match("api/users/123")
        assert pattern.match("any/arbitrary/url")
        print(f"  Matches all URLs: YES (BUG!)")

# Run the test
test_slash_only_prefix_creates_overly_broad_pattern()