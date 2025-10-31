#!/usr/bin/env python3
"""Test the proposed fix for the BrokenLinkEmailsMiddleware bug"""

import sys
from urllib.parse import urlsplit

# Simulate the fixed version of is_ignorable_request
def is_ignorable_request_fixed(uri, domain, referer):
    """Fixed version of the APPEND_SLASH check"""
    # Original broken logic:
    # if uri.endswith("/") and referer == uri[:-1]:
    #     return True

    # Fixed logic:
    if uri.endswith("/"):
        parsed_referer = urlsplit(referer)
        if (parsed_referer.netloc in ["", domain] and
            parsed_referer.path == uri[:-1]):
            return True
    return False

# Test cases
test_cases = [
    # (uri, domain, referer, expected_result, description)
    ("/page/", "example.com", "http://example.com/page", True, "Standard APPEND_SLASH redirect"),
    ("/foo/bar/", "example.com", "https://example.com/foo/bar", True, "HTTPS APPEND_SLASH redirect"),
    ("/", "example.com", "http://example.com", True, "Root path APPEND_SLASH redirect"),
    ("/page/", "example.com", "http://other.com/page", False, "Different domain - should not be ignorable"),
    ("/page/", "example.com", "http://example.com/other", False, "Different path - should not be ignorable"),
    ("/page/", "sub.example.com", "http://sub.example.com/page", True, "Subdomain APPEND_SLASH redirect"),
    ("/test/", "example.com", "/test", True, "Relative referer (no scheme/domain)"),
    ("/page", "example.com", "http://example.com/page", False, "URI doesn't end with slash - not APPEND_SLASH case"),
]

print("Testing the proposed fix:")
print("=" * 80)

all_pass = True
for uri, domain, referer, expected, description in test_cases:
    result = is_ignorable_request_fixed(uri, domain, referer)
    status = "✓" if result == expected else "✗"

    if result != expected:
        all_pass = False

    print(f"{status} {description}")
    print(f"  URI: {uri}, Domain: {domain}, Referer: {referer}")
    print(f"  Expected: {expected}, Got: {result}")
    if result != expected:
        parsed = urlsplit(referer)
        print(f"  Debug: parsed.netloc={parsed.netloc!r}, parsed.path={parsed.path!r}, uri[:-1]={uri[:-1]!r}")
    print()

print("=" * 80)
if all_pass:
    print("✓ All tests passed! The fix works correctly.")
else:
    print("✗ Some tests failed.")

# Also test with the Hypothesis examples
print("\n" + "=" * 80)
print("Testing with Hypothesis-style inputs:")
print("=" * 80)

from hypothesis import given, strategies as st, settings as hyp_settings
from hypothesis import Phase

@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=20).filter(lambda s: '/' in s),
    st.sampled_from(["http", "https"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-", min_size=4, max_size=20).filter(lambda s: '.' in s)
)
@hyp_settings(max_examples=20, phases=[Phase.generate, Phase.target])
def test_fixed_version(path, scheme, domain):
    if not path.startswith('/'):
        path = '/' + path
    if not path.endswith('/'):
        path = path + '/'

    uri = path
    referer = f"{scheme}://{domain}{path[:-1]}"

    result = is_ignorable_request_fixed(uri, domain, referer)

    assert result == True, (
        f"Fixed version should handle APPEND_SLASH correctly. "
        f"URI: {uri}, Referer: {referer}, Domain: {domain}"
    )

try:
    test_fixed_version()
    print("✓ All Hypothesis tests passed with the fixed version!")
except AssertionError as e:
    print(f"✗ Fixed version failed: {e}")