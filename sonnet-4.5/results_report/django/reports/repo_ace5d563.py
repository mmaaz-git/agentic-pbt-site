from django.middleware.common import BrokenLinkEmailsMiddleware
from django.conf import settings

# Configure Django settings
settings.configure(
    DEBUG=False,
    APPEND_SLASH=True,
    IGNORABLE_404_URLS=[],
)

# Create the middleware instance
middleware = BrokenLinkEmailsMiddleware(lambda r: r)

# Mock request object
class MockRequest:
    pass

request = MockRequest()

# Test case parameters
domain = "example.com"
uri = "/page/"
referer = "http://example.com/page"

# Call the method that has the bug
result = middleware.is_ignorable_request(request, uri, domain, referer)

# Output the results
print(f"Testing BrokenLinkEmailsMiddleware.is_ignorable_request")
print(f"=" * 60)
print(f"Input parameters:")
print(f"  domain: {domain}")
print(f"  uri: {uri}")
print(f"  referer: {referer}")
print(f"")
print(f"Expected behavior:")
print(f"  This should be an internal APPEND_SLASH redirect")
print(f"  The referer 'http://example.com/page' redirected to '/page/'")
print(f"  This should be ignorable (return True)")
print(f"")
print(f"Actual result: {result}")
print(f"Expected result: True")
print(f"")
print(f"Bug explanation:")
print(f"  The code compares referer == uri[:-1]")
print(f"  Which is: '{referer}' == '{uri[:-1]}'")
print(f"  A full URL can never equal a path-only string")