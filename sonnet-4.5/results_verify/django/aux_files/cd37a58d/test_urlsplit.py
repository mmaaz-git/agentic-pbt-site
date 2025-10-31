from urllib.parse import urlsplit

test_cases = [
    '://example.com',
    'example.com://foo',
    'https://example.com',
    'example.com'
]

for origin in test_cases:
    parsed = urlsplit(origin)
    print(f"Origin: '{origin}'")
    print(f"  scheme: '{parsed.scheme}'")
    print(f"  netloc: '{parsed.netloc}'")
    print(f"  path: '{parsed.path}'")
    print()