#!/usr/bin/env python3
"""Test reproduction for XSS vulnerability in FastAPI openapi.docs"""

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

# Test 1: XSS via title parameter in get_swagger_ui_html
print("=" * 60)
print("Test 1: XSS via title parameter in get_swagger_ui_html")
print("=" * 60)

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="</title><script>alert('XSS')</script>",
    oauth2_redirect_url=None
)

content = html.body.decode('utf-8')
print(f"Does HTML contain unescaped <script>alert?: {'<script>alert' in content}")
print(f"Does HTML contain escaped version?: {'&lt;script&gt;' in content}")

# Show the relevant part of the HTML
title_start = content.find('<title>')
title_end = content.find('</head>')
if title_start != -1 and title_end != -1:
    print("\nRelevant HTML snippet:")
    print(content[title_start:title_end])

# Test 2: XSS via oauth2_redirect_url
print("\n" + "=" * 60)
print("Test 2: XSS via oauth2_redirect_url parameter")
print("=" * 60)

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test",
    oauth2_redirect_url="'/><script>alert('OAuth2 XSS')</script>"
)

content = html.body.decode('utf-8')
print(f"Does HTML contain unescaped script injection?: {'<script>alert' in content}")

# Find oauth2RedirectUrl line
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'oauth2RedirectUrl' in line:
        print(f"\noauth2RedirectUrl line:")
        print(f"  {line}")
        break

# Test 3: JavaScript injection via openapi_url
print("\n" + "=" * 60)
print("Test 3: JavaScript injection via openapi_url")
print("=" * 60)

html = get_swagger_ui_html(
    openapi_url="javascript:alert('URL XSS')",
    title="Test",
    oauth2_redirect_url=None
)

content = html.body.decode('utf-8')
print(f"Does HTML contain javascript: protocol?: {'javascript:alert' in content}")

# Find url line
lines = content.split('\n')
for i, line in enumerate(lines):
    if "url:" in line and "openapi" not in line.lower():
        print(f"\nURL line in JavaScript:")
        print(f"  {line}")
        break

# Test 4: XSS in ReDoc HTML via title
print("\n" + "=" * 60)
print("Test 4: XSS via title parameter in get_redoc_html")
print("=" * 60)

html = get_redoc_html(
    openapi_url="/openapi.json",
    title="</title><script>alert('ReDoc XSS')</script>"
)

content = html.body.decode('utf-8')
print(f"Does HTML contain unescaped <script>alert?: {'<script>alert' in content}")

# Show the relevant part of the HTML
title_start = content.find('<title>')
title_end = content.find('</head>')
if title_start != -1 and title_end != -1:
    print("\nRelevant HTML snippet:")
    print(content[title_start:title_end])

# Test 5: Attribute injection in ReDoc via openapi_url
print("\n" + "=" * 60)
print("Test 5: Attribute injection via openapi_url in ReDoc")
print("=" * 60)

html = get_redoc_html(
    openapi_url='"><script>alert("ReDoc URL XSS")</script>',
    title="Test"
)

content = html.body.decode('utf-8')
print(f"Does HTML contain script injection?: {'<script>alert' in content}")

# Find redoc element
lines = content.split('\n')
for i, line in enumerate(lines):
    if '<redoc' in line.lower():
        print(f"\nReDoc element:")
        print(f"  {line}")
        break