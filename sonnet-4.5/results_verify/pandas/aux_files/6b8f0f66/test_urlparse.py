from urllib.parse import urlparse

print('Testing various malformed URLs:')
test_urls = ['http://[', 'http://]', 'http://[:', 'http://[::1]', 'ftp://[', 'https://[::']

for url in test_urls:
    print(f'\nURL: {url}')
    try:
        result = urlparse(url)
        print(f'  Result: {result}')
    except Exception as e:
        print(f'  Exception: {type(e).__name__}: {e}')