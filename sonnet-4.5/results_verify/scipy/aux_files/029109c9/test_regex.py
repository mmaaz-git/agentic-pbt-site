import re

r_empty = re.compile(r'^\s+$')

test_cases = ['', ' ', '  ', '\t', '\n']

for test in test_cases:
    if r_empty.match(test):
        print(f"Matches: '{repr(test)}'")
    else:
        print(f"Does NOT match: '{repr(test)}'")