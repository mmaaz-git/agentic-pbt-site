import numpy as np

# Test how Python represents complex numbers
test_values = [
    complex(1, 2),
    complex(0, 2),
    complex(1, 0),
    complex(-1, 2),
    complex(1, -2),
    complex(-1, -2),
    complex(0, 0),
    complex(3.14159, 2.71828),
    1+2j,
    2j,
    -2j,
]

print("Python complex number string representations:")
for v in test_values:
    print(f"  {v!r:20} -> str: '{str(v)}'")

print("\nNumpy complex representations:")
np_values = np.array(test_values)
for v in np_values:
    print(f"  {v!r:20} -> str: '{str(v)}'")

# Check if parentheses are always there:
print("\nDoes str() always add parentheses for complex with real+imag parts?")
for r in [-1, 0, 1, 3.5]:
    for i in [-2, 0, 2, 4.5]:
        c = complex(r, i)
        s = str(c)
        has_parens = s.startswith('(') and s.endswith(')')
        print(f"  complex({r}, {i}) = '{s}' -> has_parens: {has_parens}")