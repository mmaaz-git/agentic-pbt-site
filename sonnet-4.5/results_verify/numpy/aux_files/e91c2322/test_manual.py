import numpy.char as char

# Test the specific case from the bug report
s = 'ῂ'
arr = char.array([s])
upper = char.upper(arr)
lower_upper = char.lower(upper)

python_upper = s.upper()
python_lower_upper = python_upper.lower()

print(f"Input: {repr(s)}")
print(f"Python:  {repr(s)} -> upper: {repr(python_upper)} -> lower: {repr(python_lower_upper)}")
print(f"NumPy:   {repr(s)} -> upper: {repr(upper[0])} -> lower: {repr(lower_upper[0])}")
print(f"NumPy result: {repr(lower_upper[0])}")
print(f"Expected:     {repr(python_lower_upper)}")
print(f"Match: {lower_upper[0] == python_lower_upper}")

# Also test the German eszett found by Hypothesis
print("\n" + "="*50)
s2 = 'ß'
arr2 = char.array([s2])
upper2 = char.upper(arr2)
lower_upper2 = char.lower(upper2)

python_upper2 = s2.upper()
python_lower_upper2 = python_upper2.lower()

print(f"Input: {repr(s2)}")
print(f"Python:  {repr(s2)} -> upper: {repr(python_upper2)} -> lower: {repr(python_lower_upper2)}")
print(f"NumPy:   {repr(s2)} -> upper: {repr(upper2[0])} -> lower: {repr(lower_upper2[0])}")
print(f"NumPy result: {repr(lower_upper2[0])}")
print(f"Expected:     {repr(python_lower_upper2)}")
print(f"Match: {lower_upper2[0] == python_lower_upper2}")