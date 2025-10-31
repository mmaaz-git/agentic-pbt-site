#!/usr/bin/env python3
"""Test script to reproduce the BrokenLinkEmailsMiddleware bug"""

import sys
import os

# Add django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# First, let's run the basic reproduction test
print("=" * 60)
print("BASIC REPRODUCTION TEST")
print("=" * 60)

from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

middleware = BrokenLinkEmailsMiddleware(lambda r: r)

class MockRequest:
    pass

request = MockRequest()
domain = "example.com"
uri = "/page/"
referer = "http://example.com/page"

result = middleware.is_ignorable_request(request, uri, domain, referer)

print(f"URI: {uri}")
print(f"Referer: {referer}")
print(f"Domain: {domain}")
print(f"Should be ignorable (expected): True")
print(f"Actually ignorable: {result}")
print()

# Let's also check what the comparison actually looks like
print("=" * 60)
print("COMPARISON DETAILS")
print("=" * 60)
print(f"uri[:-1] = {uri[:-1]!r}")
print(f"referer = {referer!r}")
print(f"Are they equal? {referer == uri[:-1]}")
print()

# Now let's run the Hypothesis test
print("=" * 60)
print("HYPOTHESIS PROPERTY-BASED TEST")
print("=" * 60)

try:
    from hypothesis import given, strategies as st, settings as hyp_settings
    from hypothesis import Phase

    @given(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=50).filter(lambda s: '/' in s),
        st.sampled_from(["http", "https"]),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-", min_size=4, max_size=30).filter(lambda s: '.' in s)
    )
    @hyp_settings(max_examples=10, phases=[Phase.generate, Phase.target])
    def test_is_ignorable_request_append_slash_internal_redirect(path, scheme, domain):
        if not path.startswith('/'):
            path = '/' + path
        if not path.endswith('/'):
            path = path + '/'

        middleware = BrokenLinkEmailsMiddleware(lambda r: r)

        class MockRequest:
            pass

        request = MockRequest()
        uri = path
        referer_full_url = f"{scheme}://{domain}{path[:-1]}"

        result = middleware.is_ignorable_request(request, uri, domain, referer_full_url)

        assert result == True, (
            f"Internal redirect from APPEND_SLASH should be ignorable, but got {result}. "
            f"URI: {uri}, Referer: {referer_full_url}"
        )

    print("Running hypothesis test with 10 examples...")
    test_is_ignorable_request_append_slash_internal_redirect()
    print("All tests passed!")

except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Error running hypothesis test: {e}")

print()
print("=" * 60)
print("ADDITIONAL TEST CASES")
print("=" * 60)

# Test a few more specific cases
test_cases = [
    ("/foo/", "http://example.com/foo", "example.com"),
    ("/bar/baz/", "https://example.com/bar/baz", "example.com"),
    ("/", "http://example.com", "example.com"),
    ("/test/path/", "https://subdomain.example.com/test/path", "subdomain.example.com"),
]

for uri, referer, domain in test_cases:
    result = middleware.is_ignorable_request(request, uri, domain, referer)
    print(f"URI: {uri:<20} Referer: {referer:<40} Domain: {domain:<25}")
    print(f"  Expected: True, Got: {result}")
    print(f"  Comparison: {referer!r} == {uri[:-1]!r} -> {referer == uri[:-1]}")
    print()