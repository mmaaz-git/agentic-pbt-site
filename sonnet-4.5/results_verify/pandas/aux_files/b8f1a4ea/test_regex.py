import re

# The current regex pattern from pandas
pattern = r"^(\S*?)([a-zA-Z%!].*)"

test_cases = [
    "1e-5pt",
    "0.00001pt",
    "1.5e-3pt",
    "1e6pt",
    "100pt",
    "1.5em",
    "50%",
]

print("Testing current regex pattern:")
print(f"Pattern: {pattern}\n")

for test in test_cases:
    match = re.match(pattern, test)
    if match:
        val, unit = match.groups()
        print(f"Input: {test:15} -> Value: '{val}', Unit: '{unit}'")
    else:
        print(f"Input: {test:15} -> No match")

print("\n" + "="*50 + "\n")

# The proposed fix regex pattern
new_pattern = r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z%!].*)"

print("Testing proposed regex pattern:")
print(f"Pattern: {new_pattern}\n")

for test in test_cases:
    match = re.match(new_pattern, test)
    if match:
        val, unit = match.groups()
        print(f"Input: {test:15} -> Value: '{val}', Unit: '{unit}'")
    else:
        print(f"Input: {test:15} -> No match")