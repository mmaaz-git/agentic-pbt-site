from Cython.Build.Dependencies import parse_list

# Test all the failing cases mentioned in the report
test_cases = [
    '[""]',           # Empty string in list
    "['']",           # Empty single-quoted string in list
    '""',             # Empty string without brackets
    "''",             # Empty single-quoted string without brackets
    '"',              # Unclosed double quote
    "'",              # Unclosed single quote
    '["\\"]',         # Escaped quote in list
    '[a, "", b]',     # Empty string in mixed list
]

for test in test_cases:
    try:
        result = parse_list(test)
        print(f"parse_list('{test}') = {result}")
    except KeyError as e:
        print(f"parse_list('{test}') -> KeyError: {e}")
    except Exception as e:
        print(f"parse_list('{test}') -> {type(e).__name__}: {e}")