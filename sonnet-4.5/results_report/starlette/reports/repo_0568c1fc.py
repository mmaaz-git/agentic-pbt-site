from starlette.middleware.errors import ServerErrorMiddleware

# Create an instance of ServerErrorMiddleware
middleware = ServerErrorMiddleware(app=lambda scope, receive, send: None)

# Test the format_line method with a string containing spaces
test_line = "hello world"
formatted = middleware.format_line(0, test_line, 1, 0)

print("Input line:", repr(test_line))
print("\nFormatted output:")
print(formatted)
print("\nSearching for '&nbsp' (without semicolon):", "&nbsp" in formatted and "&nbsp;" not in formatted)
print("Searching for '&nbsp;' (with semicolon):", "&nbsp;" in formatted)