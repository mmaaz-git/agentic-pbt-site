from starlette.datastructures import URL

# Test case that causes the crash
url = URL("http://@/path")
result = url.replace(port=8000)
print(f"Result: {result}")