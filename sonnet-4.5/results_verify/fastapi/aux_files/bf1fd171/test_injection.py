"""Test for HTML/JavaScript injection vulnerabilities in FastAPI OpenAPI docs"""
from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Test 1: Property-based test for title HTML injection
@given(title=st.text())
@settings(max_examples=100)
def test_title_should_not_allow_html_injection(title):
    result = get_swagger_ui_html(
        openapi_url="http://example.com/openapi.json",
        title=title,
    )
    html = result.body.decode('utf-8')

    if '<script>' in title.lower() or '</title>' in title.lower():
        # Check if the script tag appears in the HTML title section
        assert '<script>' not in html or title not in html.split('<title>')[1].split('</title>')[0]

# Test 2: Property-based test for openapi_url JavaScript injection
@given(openapi_url=st.text())
@settings(max_examples=100)
def test_openapi_url_should_not_allow_js_injection(openapi_url):
    result = get_swagger_ui_html(
        openapi_url=openapi_url,
        title="Test",
    )
    html = result.body.decode('utf-8')

    if "'; alert(" in openapi_url:
        assert "'; alert(" not in html or openapi_url not in html

# Manual reproduction tests
def test_manual_reproductions():
    print("\n=== Manual Reproduction Tests ===\n")

    # Test 1: Title HTML injection in swagger_ui_html
    print("Test 1: Title HTML injection in swagger_ui_html")
    malicious_title = "</title><script>alert('XSS')</script><title>Fake"
    result = get_swagger_ui_html(
        openapi_url="http://example.com/openapi.json",
        title=malicious_title,
    )
    html = result.body.decode('utf-8')

    if "<script>alert('XSS')</script>" in html:
        print(f"  ✗ VULNERABLE: Script tag found in HTML")
        # Extract the title section to show the issue
        title_start = html.find('<title>')
        title_end = html.find('</head>')
        print(f"  HTML snippet: {html[title_start:title_end][:200]}...")
    else:
        print(f"  ✓ SAFE: Script tag not found in HTML")

    # Test 2: OpenAPI URL JavaScript injection
    print("\nTest 2: OpenAPI URL JavaScript injection")
    malicious_url = "'; alert('XSS'); //'"
    result = get_swagger_ui_html(
        openapi_url=malicious_url,
        title="API",
    )
    html = result.body.decode('utf-8')

    if "alert('XSS')" in html:
        print(f"  ✗ VULNERABLE: JavaScript injection found")
        # Find the script section where the URL is used
        script_start = html.find("url: '")
        if script_start != -1:
            print(f"  JS snippet: {html[script_start:script_start+100]}...")
    else:
        print(f"  ✓ SAFE: JavaScript injection not found")

    # Test 3: Title HTML injection in redoc_html
    print("\nTest 3: Title HTML injection in redoc_html")
    result = get_redoc_html(
        openapi_url="http://example.com/openapi.json",
        title=malicious_title,
    )
    html = result.body.decode('utf-8')

    if "<script>alert('XSS')</script>" in html:
        print(f"  ✗ VULNERABLE: Script tag found in ReDoc HTML")
        title_start = html.find('<title>')
        title_end = html.find('</title>') + 8
        print(f"  HTML snippet: {html[title_start:title_end]}")
    else:
        print(f"  ✓ SAFE: Script tag not found in ReDoc HTML")

    # Test 4: OAuth2 redirect URL injection
    print("\nTest 4: OAuth2 redirect URL injection")
    malicious_oauth = "'; alert('OAuth XSS'); //'"
    result = get_swagger_ui_html(
        openapi_url="http://example.com/openapi.json",
        title="Test",
        oauth2_redirect_url=malicious_oauth
    )
    html = result.body.decode('utf-8')

    if "alert('OAuth XSS')" in html:
        print(f"  ✗ VULNERABLE: OAuth redirect injection found")
        oauth_pos = html.find("oauth2RedirectUrl")
        if oauth_pos != -1:
            print(f"  JS snippet: {html[oauth_pos:oauth_pos+100]}...")
    else:
        print(f"  ✓ SAFE: OAuth redirect injection not found")

    # Test 5: CSS URL attribute injection
    print("\nTest 5: CSS URL attribute injection")
    malicious_css = '"><script>alert("CSS")</script><link href="'
    result = get_swagger_ui_html(
        openapi_url="http://example.com/openapi.json",
        title="Test",
        swagger_css_url=malicious_css
    )
    html = result.body.decode('utf-8')

    if '<script>alert("CSS")</script>' in html:
        print(f"  ✗ VULNERABLE: CSS URL injection found")
        css_pos = html.find('rel="stylesheet"')
        if css_pos != -1:
            print(f"  HTML snippet: {html[css_pos-50:css_pos+100]}...")
    else:
        print(f"  ✓ SAFE: CSS URL injection not found")

if __name__ == "__main__":
    print("Running manual reproduction tests...")
    test_manual_reproductions()

    print("\n\nRunning hypothesis property-based tests...")
    try:
        test_title_should_not_allow_html_injection()
        print("Title injection property test: No failures detected in sampling")
    except AssertionError as e:
        print(f"Title injection property test: FAILED - {e}")

    try:
        test_openapi_url_should_not_allow_js_injection()
        print("OpenAPI URL injection property test: No failures detected in sampling")
    except AssertionError as e:
        print(f"OpenAPI URL injection property test: FAILED - {e}")