#!/usr/bin/env python3
"""Property-based test for XSS vulnerability using Hypothesis"""

from hypothesis import given, strategies as st, settings, assume
from fastapi.openapi.docs import get_swagger_ui_html

@given(
    title=st.text(min_size=1, max_size=100),
    openapi_url=st.text(min_size=1, max_size=200),
    oauth2_redirect_url=st.one_of(st.none(), st.text(min_size=1, max_size=200))
)
@settings(max_examples=50)
def test_no_xss_injection(title, openapi_url, oauth2_redirect_url):
    """HTML/JS should be properly escaped to prevent XSS"""
    html = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        oauth2_redirect_url=oauth2_redirect_url
    )
    html_content = html.body.decode('utf-8')

    # Check if there's a script tag injection
    if "</script>" in title or "</script>" in openapi_url or (oauth2_redirect_url and "</script>" in oauth2_redirect_url):
        # If user input contains </script>, it should either not appear in output or be escaped
        if "</script>" in html_content:
            # Count occurrences - there should only be the legitimate script tags, not from user input
            # We expect 2 legitimate </script> tags in the template
            legitimate_count = 2  # One for swagger-ui-bundle.js, one for the inline script
            actual_count = html_content.count("</script>")

            # Additional scripts from malicious input would increase the count
            if actual_count > legitimate_count:
                print(f"\nFAILURE: XSS vulnerability found!")
                print(f"  title: {repr(title)}")
                print(f"  openapi_url: {repr(openapi_url)}")
                print(f"  oauth2_redirect_url: {repr(oauth2_redirect_url)}")
                print(f"  Expected <=2 </script> tags, found {actual_count}")
                assert False, "XSS injection detected - unescaped </script> tag in output"

    # Check for javascript: protocol injection
    if "javascript:" in openapi_url:
        if "javascript:" in html_content:
            # The javascript: should not appear in a context where it can be executed
            # It should especially not appear inside url: '...' in the JavaScript
            lines = html_content.split('\n')
            for line in lines:
                if "url:" in line and "javascript:" in line:
                    print(f"\nFAILURE: JavaScript protocol injection found!")
                    print(f"  openapi_url: {repr(openapi_url)}")
                    print(f"  Dangerous line: {line.strip()}")
                    assert False, "XSS injection detected - javascript: protocol in URL"

# Manual testing helper function
def test_specific_input(title, openapi_url, oauth2_redirect_url):
    """Test specific inputs manually"""
    html = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        oauth2_redirect_url=oauth2_redirect_url
    )
    html_content = html.body.decode('utf-8')

    # Check if there's a script tag injection
    if "</script>" in title or "</script>" in openapi_url or (oauth2_redirect_url and "</script>" in oauth2_redirect_url):
        # If user input contains </script>, it should either not appear in output or be escaped
        if "</script>" in html_content:
            # Count occurrences - there should only be the legitimate script tags, not from user input
            # We expect 2 legitimate </script> tags in the template
            legitimate_count = 2  # One for swagger-ui-bundle.js, one for the inline script
            actual_count = html_content.count("</script>")

            # Additional scripts from malicious input would increase the count
            if actual_count > legitimate_count:
                print(f"\nFAILURE: XSS vulnerability found!")
                print(f"  title: {repr(title)}")
                print(f"  openapi_url: {repr(openapi_url)}")
                print(f"  oauth2_redirect_url: {repr(oauth2_redirect_url)}")
                print(f"  Expected <=2 </script> tags, found {actual_count}")
                return False

    # Check for javascript: protocol injection
    if "javascript:" in openapi_url:
        if "javascript:" in html_content:
            # The javascript: should not appear in a context where it can be executed
            # It should especially not appear inside url: '...' in the JavaScript
            lines = html_content.split('\n')
            for line in lines:
                if "url:" in line and "javascript:" in line:
                    print(f"\nFAILURE: JavaScript protocol injection found!")
                    print(f"  openapi_url: {repr(openapi_url)}")
                    print(f"  Dangerous line: {line.strip()}")
                    return False

    return True

# Test with specific known-bad inputs
print("Testing with specific malicious inputs...")

# Test case 1: Script injection via title
result = test_specific_input(
    title="</title><script>alert(1)</script>",
    openapi_url="/openapi.json",
    oauth2_redirect_url=None
)
if result:
    print("✗ Test case 1 (title injection): Passed (unexpected - vulnerability not detected)")
else:
    print("✓ Test case 1 (title injection): Failed as expected - vulnerability detected")

# Test case 2: JavaScript protocol injection
result = test_specific_input(
    title="Test",
    openapi_url="javascript:alert(1)",
    oauth2_redirect_url=None
)
if result:
    print("✗ Test case 2 (javascript: protocol): Passed (unexpected - vulnerability not detected)")
else:
    print("✓ Test case 2 (javascript: protocol): Failed as expected - vulnerability detected")

# Test case 3: OAuth2 redirect URL injection
result = test_specific_input(
    title="Test",
    openapi_url="/openapi.json",
    oauth2_redirect_url="'/><script>alert(1)</script>"
)
if result:
    print("✗ Test case 3 (oauth2 injection): Passed (unexpected - vulnerability not detected)")
else:
    print("✓ Test case 3 (oauth2 injection): Failed as expected - vulnerability detected")

print("\nRunning random property-based tests...")
try:
    test_no_xss_injection()
    print("Property-based test passed (no vulnerabilities detected in random inputs)")
except AssertionError as e:
    print(f"Property-based test failed: {e}")