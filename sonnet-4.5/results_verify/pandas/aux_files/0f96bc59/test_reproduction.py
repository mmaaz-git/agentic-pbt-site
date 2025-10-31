import pandas.util

# Test 1: German sharp s
result1 = pandas.util.capitalize_first_letter('ß')
print(f"capitalize_first_letter('ß') = {repr(result1)}")
print(f"Expected: 'SS', Got: {repr(result1)}, Match: {result1 == 'SS'}")
print(f"Length of 'ß': {len('ß')}, Length of result: {len(result1)}")
print()

# Test 2: ßeta
result2 = pandas.util.capitalize_first_letter('ßeta')
print(f"capitalize_first_letter('ßeta') = {repr(result2)}")
print(f"Expected: 'SSeta', Got: {repr(result2)}, Match: {result2 == 'SSeta'}")
print(f"'ßeta'[1:] = {repr('ßeta'[1:])}, 'SSeta'[1:] = {repr('SSeta'[1:])}")
print()

# Test 3: ligature fi
result3 = pandas.util.capitalize_first_letter('ﬁle')
print(f"capitalize_first_letter('ﬁle') = {repr(result3)}")
print(f"Expected: 'FIle', Got: {repr(result3)}, Match: {result3 == 'FIle'}")
print(f"Length of 'ﬁle': {len('ﬁle')}, Length of result: {len(result3)}")

# Verify assertions from bug report
try:
    assert pandas.util.capitalize_first_letter('ß') == 'SS'
    assert len('ß') == 1 and len('SS') == 2
    assert pandas.util.capitalize_first_letter('ßeta') == 'SSeta'
    assert 'ßeta'[1:] == 'eta' and 'SSeta'[1:] == 'Seta'
    assert pandas.util.capitalize_first_letter('ﬁle') == 'FIle'
    assert len('ﬁle') == 3 and len('FIle') == 4
    print("\nAll assertions from bug report passed!")
except AssertionError as e:
    print(f"\nAssertion failed: {e}")