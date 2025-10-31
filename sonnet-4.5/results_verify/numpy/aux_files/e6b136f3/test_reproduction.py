import numpy.char as char
import numpy as np

test_char = 'ß'
arr = np.array([test_char])

print(f'Testing: {test_char!r}')
print(f'upper:      numpy={char.upper(arr)[0]!r:5} Python={test_char.upper()!r}')
print(f'capitalize: numpy={char.capitalize(arr)[0]!r:5} Python={test_char.capitalize()!r}')
print(f'title:      numpy={char.title(arr)[0]!r:5} Python={test_char.title()!r}')
print(f'swapcase:   numpy={char.swapcase(arr)[0]!r:5} Python={test_char.swapcase()!r}')

# Test with another character that expands
test_char2 = 'ﬁ'  # Latin Small Ligature Fi (U+FB01)
arr2 = np.array([test_char2])

print(f'\nTesting: {test_char2!r}')
print(f'upper:      numpy={char.upper(arr2)[0]!r:5} Python={test_char2.upper()!r}')
print(f'capitalize: numpy={char.capitalize(arr2)[0]!r:5} Python={test_char2.capitalize()!r}')
print(f'title:      numpy={char.title(arr2)[0]!r:5} Python={test_char2.title()!r}')
print(f'swapcase:   numpy={char.swapcase(arr2)[0]!r:5} Python={test_char2.swapcase()!r}')