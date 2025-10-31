from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = ServerErrorMiddleware(dummy_app, debug=True)
test_line = "    some code with spaces"
formatted = middleware.format_line(5, test_line, 10, 5)

print("Formatted output:")
print(formatted)
print("\n--- Analysis ---")
if "&nbsp;" in formatted:
    print("✓ Correct: HTML entities end with semicolon")
elif "&nbsp" in formatted:
    print("✗ Bug confirmed: HTML entities missing semicolon")
    print(f"Found {formatted.count('&nbsp')} instances of '&nbsp' without semicolon")