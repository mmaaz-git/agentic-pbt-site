import numpy.char as char
import numpy as np

test_cases = [
    ('ß', 'Ss'),
    ('ﬁ', 'Fi'),
    ('ﬂ', 'Fl'),
    ('ﬀ', 'Ff'),
]

print("Testing individual cases:")
for s, expected_python in test_cases:
    arr = np.array([s])
    numpy_result = char.title(arr)[0]
    python_result = s.title()

    print(f'{s!r}: numpy={numpy_result!r}, Python={python_result!r}, match={numpy_result == python_result}')

print("\nSpecific test cases:")
arr = np.array(['ß'])
print(f'char.title(["ß"]) = {char.title(arr)[0]!r} (expected: "Ss")')

arr2 = np.array(['ﬁ'])
print(f'char.title(["ﬁ"]) = {char.title(arr2)[0]!r} (expected: "Fi")')

# Additional test to verify Python's behavior
print("\nVerifying Python str.title() behavior:")
print(f'"ß".title() = {"ß".title()!r}')
print(f'"ﬁ".title() = {"ﬁ".title()!r}')
print(f'"ﬂ".title() = {"ﬂ".title()!r}')
print(f'"ﬀ".title() = {"ﬀ".title()!r}')