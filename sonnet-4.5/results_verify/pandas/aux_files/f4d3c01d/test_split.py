import re

# Test what the split does with parentheses
test_inputs = ["(1+2j)", "1+2j", "(3-4j)", "-3+4j", "(5j)", "5j"]

for x in test_inputs:
    trimmed = re.split(r"([j+-])", x)
    print(f"Input: {x:10} -> Split: {trimmed}")

    # What the current code extracts:
    real_part = "".join(trimmed[:-4])
    imag_part = "".join(trimmed[-4:-2])
    print(f"  Real: '{real_part}', Imag: '{imag_part}'")
    print()