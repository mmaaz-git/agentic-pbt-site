"""Run the Hypothesis test from the bug report"""
import django
from django.conf import settings

# Configure Django
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test',
        CSRF_TRUSTED_ORIGINS=[]
    )
    django.setup()

from hypothesis import given, strategies as st
from unittest.mock import patch
from urllib.parse import urlsplit
from django.core.checks.compatibility.django_4_0 import check_csrf_trusted_origins

# Test the specific failing input mentioned
print("Testing the specific failing input from bug report:")
prefix = 'a'
suffix = 'a'
origin = f"{prefix}://{suffix}"
parsed = urlsplit(origin)
is_valid_scheme = parsed.scheme in ['http', 'https', 'ftp', 'ws', 'wss']

print(f"Origin: '{origin}'")
print(f"Parsed scheme: '{parsed.scheme}', netloc: '{parsed.netloc}'")
print(f"Is valid scheme (according to test): {is_valid_scheme}")

with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
    result = check_csrf_trusted_origins(app_configs=None)
    print(f"Errors returned: {len(result)}")
    if not is_valid_scheme and len(result) == 0:
        print("BUG CONFIRMED: Origin with invalid scheme 'a' was accepted")

# Run the property test
@given(
    st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(
        lambda s: "://" not in s
    ),
    st.text(min_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)).filter(
        lambda s: "://" not in s
    )
)
def test_scheme_must_be_at_start_not_anywhere(prefix, suffix):
    origin = f"{prefix}://{suffix}"
    parsed = urlsplit(origin)
    is_valid_scheme = parsed.scheme in ['http', 'https', 'ftp', 'ws', 'wss']

    with patch.object(settings, 'CSRF_TRUSTED_ORIGINS', [origin]):
        result = check_csrf_trusted_origins(app_configs=None)
        if not is_valid_scheme:
            assert len(result) > 0, f"Origin '{origin}' should be rejected but wasn't"

print("\nRunning property-based test...")
try:
    test_scheme_must_be_at_start_not_anywhere()
    print("Property test passed (no assertion errors)")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Error during test: {e}")