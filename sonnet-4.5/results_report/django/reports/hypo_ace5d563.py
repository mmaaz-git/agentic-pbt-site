from hypothesis import given, strategies as st
from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=2, max_size=50).filter(lambda s: '/' in s),
    st.sampled_from(["http", "https"]),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz.-", min_size=4, max_size=30).filter(lambda s: '.' in s)
)
def test_is_ignorable_request_append_slash_internal_redirect(path, scheme, domain):
    # Ensure path starts with / and ends with /
    if not path.startswith('/'):
        path = '/' + path
    if not path.endswith('/'):
        path = path + '/'

    # Create middleware instance
    middleware = BrokenLinkEmailsMiddleware(lambda r: r)

    # Mock request object
    class MockRequest:
        pass

    request = MockRequest()

    # Set up test parameters
    uri = path
    referer_full_url = f"{scheme}://{domain}{path[:-1]}"

    # Call the method
    result = middleware.is_ignorable_request(request, uri, domain, referer_full_url)

    # Assert the expected behavior
    assert result == True, (
        f"Internal redirect from APPEND_SLASH should be ignorable, but got {result}. "
        f"URI: {uri}, Referer: {referer_full_url}"
    )

# Run the test
if __name__ == "__main__":
    test_is_ignorable_request_append_slash_internal_redirect()