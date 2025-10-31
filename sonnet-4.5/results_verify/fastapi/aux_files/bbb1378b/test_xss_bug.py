#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# First, run the hypothesis test
@settings(max_examples=500)
@given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
def test_swagger_ui_html_escapes_title_properly(openapi_url, title):
    response = get_swagger_ui_html(openapi_url=openapi_url, title=title)
    content = response.body.decode('utf-8')

    title_tag_start = content.find('<title>')
    title_tag_end = content.find('</title>')

    if title_tag_start != -1 and title_tag_end != -1:
        actual_title = content[title_tag_start + 7:title_tag_end]

        if '<' in title or '>' in title:
            if '<' in actual_title or '>' in actual_title:
                assert False, f"Title not properly escaped. Got {actual_title!r} from input {title!r}"

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_swagger_ui_html_escapes_title_properly()
    print("Hypothesis test passed - this is unexpected!")
except AssertionError as e:
    print(f"Hypothesis test failed as expected: {e}")

# Manual reproduction tests
print("\n" + "="*60)
print("Manual reproduction tests:")
print("="*60)

# Test 1: XSS in title
xss_title = '<script>alert("XSS")</script>'
print(f"\nTest 1: XSS in title")
print(f"Input title: {xss_title}")

response = get_swagger_ui_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

if xss_title in content:
    print("✗ XSS vulnerability confirmed: Raw script tag found in HTML output")
    # Find where it appears
    idx = content.find(xss_title)
    print(f"  Found at position {idx}: {content[max(0,idx-20):idx+len(xss_title)+20]}")
else:
    print("✓ XSS not found - title is properly escaped")

# Test 2: XSS in openapi_url
xss_url = '"><script>alert("XSS")</script>'
print(f"\nTest 2: XSS in openapi_url")
print(f"Input URL: {xss_url}")

response = get_swagger_ui_html(openapi_url=xss_url, title="Test")
content = response.body.decode('utf-8')

if 'alert("XSS")' in content:
    print("✗ XSS vulnerability confirmed: Script found in HTML output")
    idx = content.find('alert("XSS")')
    print(f"  Found at position {idx}: {content[max(0,idx-20):idx+20]}")
else:
    print("✓ XSS not found - URL is properly escaped")

# Test 3: XSS in redoc title
print(f"\nTest 3: XSS in ReDoc title")
print(f"Input title: {xss_title}")

response = get_redoc_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

if xss_title in content:
    print("✗ XSS vulnerability confirmed in ReDoc: Raw script tag found in HTML output")
    idx = content.find(xss_title)
    print(f"  Found at position {idx}: {content[max(0,idx-20):idx+len(xss_title)+20]}")
else:
    print("✓ XSS not found in ReDoc - title is properly escaped")

# Test 4: XSS in redoc openapi_url
print(f"\nTest 4: XSS in ReDoc openapi_url")
print(f"Input URL: {xss_url}")

response = get_redoc_html(openapi_url=xss_url, title="Test")
content = response.body.decode('utf-8')

if 'alert("XSS")' in content:
    print("✗ XSS vulnerability confirmed in ReDoc: Script found in HTML output")
    idx = content.find('alert("XSS")')
    print(f"  Found at position {idx}: {content[max(0,idx-20):idx+20]}")
else:
    print("✓ XSS not found in ReDoc - URL is properly escaped")

# Test 5: Less obvious XSS with simple angle brackets
print(f"\nTest 5: Simple angle brackets in title")
simple_title = 'Test < and > Characters'
response = get_swagger_ui_html(openapi_url="/openapi.json", title=simple_title)
content = response.body.decode('utf-8')

title_tag_start = content.find('<title>')
title_tag_end = content.find('</title>')
if title_tag_start != -1 and title_tag_end != -1:
    actual_title = content[title_tag_start + 7:title_tag_end]
    print(f"Input: {simple_title}")
    print(f"Output in <title> tag: {actual_title}")
    if actual_title == simple_title:
        print("✗ Raw angle brackets preserved - potential for HTML injection")
    else:
        print("✓ Angle brackets properly escaped")