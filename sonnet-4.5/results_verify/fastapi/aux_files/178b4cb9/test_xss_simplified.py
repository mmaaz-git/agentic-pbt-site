#!/usr/bin/env python3
"""Simplified XSS tests"""

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Test the most basic XSS case
def test_basic_xss():
    print("Testing basic XSS scenarios...")

    # Test 1: Simple < character in title (mentioned as failing input)
    title = '<'
    result = get_swagger_ui_html(
        openapi_url="https://example.com/openapi.json",
        title=title
    )
    html = result.body.decode('utf-8')
    print(f"\n1. Title with '<': ")
    print(f"   Input: {repr(title)}")
    print(f"   In HTML: <title>{title}</title> appears as-is: {f'<title>{title}</title>' in html}")

    # Test 2: Script tag
    title = '<script>alert(1)</script>'
    result = get_swagger_ui_html(
        openapi_url="https://example.com/openapi.json",
        title=title
    )
    html = result.body.decode('utf-8')
    print(f"\n2. Script tag in title:")
    print(f"   Input: {repr(title)}")
    print(f"   Appears unescaped: {title in html}")

    # Test 3: Quote in URL for JavaScript context
    url = "'; alert(1); //'"
    result = get_swagger_ui_html(
        openapi_url=url,
        title="Test"
    )
    html = result.body.decode('utf-8')
    print(f"\n3. Quote in URL (JavaScript context):")
    print(f"   Input: {repr(url)}")
    print(f"   In JS: url: '{url}'")
    js_string = f"url: '{url}'"
    print(f"   Breaks out of string: {js_string in html}")

    # Test 4: Double quote in ReDoc URL
    url = '" onload="alert(1)"'
    result = get_redoc_html(
        openapi_url=url,
        title="Test"
    )
    html = result.body.decode('utf-8')
    print(f"\n4. Double quote in ReDoc URL (HTML attribute):")
    print(f"   Input: {repr(url)}")
    print(f"   In HTML: <redoc spec-url=\"{url}\">")
    print(f"   Breaks out of attribute: {f'spec-url=\"{url}\"' in html}")

    # Print actual output samples
    print("\n=== Actual HTML samples ===")

    # Show a vulnerable title
    result = get_swagger_ui_html(
        openapi_url="https://example.com/openapi.json",
        title="<img src=x onerror=alert(1)>"
    )
    html = result.body.decode('utf-8')
    import re
    title_match = re.search(r'<title>.*?</title>', html, re.DOTALL)
    if title_match:
        print(f"Swagger UI title tag: {title_match.group()}")

    # Show vulnerable JS context
    result = get_swagger_ui_html(
        openapi_url="https://example.com';alert(1);//",
        title="Test"
    )
    html = result.body.decode('utf-8')
    url_match = re.search(r"url: '[^']*'", html)
    if url_match:
        print(f"Swagger UI JavaScript: {url_match.group()}")

    # Show vulnerable ReDoc HTML attribute
    result = get_redoc_html(
        openapi_url='https://example.com"><script>alert(1)</script>',
        title="Test"
    )
    html = result.body.decode('utf-8')
    spec_match = re.search(r'<redoc spec-url="[^"]*"', html)
    if spec_match:
        print(f"ReDoc HTML attribute: {spec_match.group()[:100]}...")

if __name__ == "__main__":
    test_basic_xss()