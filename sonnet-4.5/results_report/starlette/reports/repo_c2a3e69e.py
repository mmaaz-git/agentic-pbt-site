from starlette.datastructures import URL

# Test case that causes IndexError
url = URL("http:///path")
print(f"Created URL: {url}")
print(f"URL components: scheme='{url.scheme}', netloc='{url.netloc}', path='{url.path}'")
print(f"Attempting to replace port...")

try:
    new_url = url.replace(port=8080)
    print(f"Success: {new_url}")
except IndexError as e:
    print(f"IndexError: {e}")
    import traceback
    traceback.print_exc()