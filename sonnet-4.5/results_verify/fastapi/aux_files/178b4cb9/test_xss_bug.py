#!/usr/bin/env python3
"""Test XSS vulnerability in FastAPI documentation functions"""

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import re

# First, let's test with the specific examples from the bug report
def test_specific_xss_examples():
    print("Testing specific XSS examples...")

    # Test 1: XSS in title for swagger UI
    title_xss = '<script>alert("XSS")</script>'
    result = get_swagger_ui_html(
        openapi_url="https://example.com/openapi.json",
        title=title_xss
    )
    html = result.body.decode('utf-8')
    print("\n=== Test 1: XSS in Swagger UI title ===")
    print(f"Input title: {title_xss}")
    if '<script>alert("XSS")</script>' in html:
        print("VULNERABLE: Script tag found unescaped in HTML")
        # Show the relevant part
        title_match = re.search(r'<title>.*?</title>', html, re.DOTALL)
        if title_match:
            print(f"Found in HTML: {title_match.group()}")
    else:
        print("NOT VULNERABLE: Script tag was escaped or not present")

    # Test 2: XSS in openapi_url for swagger UI (JavaScript context)
    url_xss = "https://example.com'; alert('XSS'); var x='"
    result2 = get_swagger_ui_html(
        openapi_url=url_xss,
        title="Safe Title"
    )
    html2 = result2.body.decode('utf-8')
    print("\n=== Test 2: XSS in Swagger UI openapi_url ===")
    print(f"Input URL: {url_xss}")
    if "url: 'https://example.com'; alert('XSS'); var x=''" in html2:
        print("VULNERABLE: Unescaped JavaScript injection found")
        # Show the relevant part
        url_match = re.search(r"url: '[^']*'", html2)
        if url_match:
            print(f"Found in JavaScript: {url_match.group()}")
    else:
        print("NOT VULNERABLE: JavaScript was escaped or not present")

    # Test 3: XSS in title for ReDoc
    redoc_title_xss = '<script>alert("XSS in ReDoc")</script>'
    result3 = get_redoc_html(
        openapi_url="https://example.com/openapi.json",
        title=redoc_title_xss
    )
    html3 = result3.body.decode('utf-8')
    print("\n=== Test 3: XSS in ReDoc title ===")
    print(f"Input title: {redoc_title_xss}")
    if '<script>alert("XSS in ReDoc")</script>' in html3:
        print("VULNERABLE: Script tag found unescaped in HTML")
        title_match = re.search(r'<title>.*?</title>', html3, re.DOTALL)
        if title_match:
            print(f"Found in HTML: {title_match.group()}")
    else:
        print("NOT VULNERABLE: Script tag was escaped or not present")

    # Test 4: XSS in openapi_url for ReDoc (HTML attribute context)
    redoc_url_xss = 'https://example.com"><script>alert("XSS")</script><redoc spec-url="'
    result4 = get_redoc_html(
        openapi_url=redoc_url_xss,
        title="Safe Title"
    )
    html4 = result4.body.decode('utf-8')
    print("\n=== Test 4: XSS in ReDoc openapi_url ===")
    print(f"Input URL: {redoc_url_xss}")
    if '"><script>alert("XSS")</script>' in html4:
        print("VULNERABLE: Unescaped HTML injection found")
        spec_match = re.search(r'<redoc spec-url="[^"]*"', html4)
        if spec_match:
            print(f"Found in HTML: {spec_match.group()}")
    else:
        print("NOT VULNERABLE: HTML was escaped or not present")

# Now test with hypothesis
@given(
    openapi_url=st.text(min_size=1, max_size=100),
    title=st.text(min_size=1, max_size=100)
)
@settings(max_examples=100)  # Reduced for testing
def test_swagger_ui_html_xss_injection_attempts(openapi_url, title):
    try:
        result = get_swagger_ui_html(openapi_url=openapi_url, title=title)
        html_content = result.body.decode('utf-8')

        # Check for unescaped dangerous characters in title
        if '<' in title or '>' in title or '"' in title:
            # Check if these appear unescaped in the HTML title tag
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.DOTALL)
            if title_match and title in title_match.group(1):
                print(f"\nPotential XSS in title: {repr(title)}")
                return False

        # Check for unescaped quotes in JavaScript context
        if "'" in openapi_url:
            # Check if single quotes appear unescaped in JavaScript string
            if f"url: '{openapi_url}'" in html_content:
                print(f"\nPotential XSS in URL (JS context): {repr(openapi_url)}")
                return False

    except Exception as e:
        # Ignore any exceptions from weird inputs
        pass

    return True

if __name__ == "__main__":
    # Run specific tests
    test_specific_xss_examples()

    # Run hypothesis tests
    print("\n=== Running Hypothesis property-based tests ===")
    failed_count = 0
    for i in range(50):  # Run a subset
        try:
            test_swagger_ui_html_xss_injection_attempts()
        except AssertionError:
            failed_count += 1

    if failed_count > 0:
        print(f"\n{failed_count} property-based test cases found potential XSS vulnerabilities")
    else:
        print("\nNo XSS vulnerabilities found in property-based tests")