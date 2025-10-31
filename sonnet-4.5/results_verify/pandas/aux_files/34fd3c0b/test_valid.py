import pandas.api.types as pat

valid_patterns = [
    ".*",
    "[a-z]+",
    "(test)",
    "\\d+",
    "^start$",
    ""  # empty string is a valid regex
]

for pattern in valid_patterns:
    result = pat.is_re_compilable(pattern)
    print(f"is_re_compilable('{pattern}'): {result}")

# Test non-string types that should return False
non_strings = [1, None, [], {}, 3.14]
for obj in non_strings:
    result = pat.is_re_compilable(obj)
    print(f"is_re_compilable({obj}): {result}")