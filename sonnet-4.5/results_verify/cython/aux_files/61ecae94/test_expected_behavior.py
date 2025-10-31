from Cython.TestUtils import _parse_pattern

# Test valid patterns that should work
test_cases = [
    ("simple_pattern", (None, None, "simple_pattern")),
    ("/start/pattern", ("start", None, "pattern")),
    ("/start/:/end/pattern", ("start", "end", "pattern")),
    (":/end/pattern", (None, "end", "pattern")),
    ("/start/ :/end/ pattern", ("start", "end", "pattern")),  # with spaces
    ("/star\\/t/pattern", ("star\\/t", None, "pattern")),  # escaped slash in start
    ("", (None, None, "")),  # empty pattern
    ("   whitespace   ", (None, None, "whitespace")),  # whitespace handling
]

print("Testing valid patterns:")
for pattern, expected in test_cases:
    try:
        result = _parse_pattern(pattern)
        success = result == expected
        print(f"Pattern: {pattern!r:30} -> {result} {'✓' if success else f'✗ Expected: {expected}'}")
    except Exception as e:
        print(f"Pattern: {pattern!r:30} -> ERROR: {e}")

print("\nTesting edge cases:")
edge_cases = [
    "/",  # single slash
    "/start",  # no closing slash
    "/start\\/",  # escaped closing slash
    ":/",  # colon with single slash
    ":/end",  # no closing slash after colon
    "/\\/escaped",  # escaped slash with no closing
    "/:/",  # slash colon slash
    "//",  # double slash
]

for pattern in edge_cases:
    try:
        result = _parse_pattern(pattern)
        print(f"Pattern: {pattern!r:20} -> {result}")
    except Exception as e:
        print(f"Pattern: {pattern!r:20} -> ERROR: {type(e).__name__}: {e}")