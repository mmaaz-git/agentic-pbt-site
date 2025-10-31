import html
from starlette.middleware.errors import ServerErrorMiddleware

# Create a middleware instance with debug enabled
middleware = ServerErrorMiddleware(None, debug=True)

# Test line with leading spaces that should be converted to HTML entities
test_line = "    def example_function():"

# This is what the format_line method does internally (line 191 of errors.py)
escaped = html.escape(test_line).replace(" ", "&nbsp")

print("Input line:")
print(repr(test_line))
print("\nOutput from the buggy code:")
print(escaped)
print("\nExpected output (with proper HTML entities):")
print(html.escape(test_line).replace(" ", "&nbsp;"))
print("\nNote: The output is missing semicolons after '&nbsp' entities")
print("This violates HTML5 specification which requires named character references to end with semicolon")