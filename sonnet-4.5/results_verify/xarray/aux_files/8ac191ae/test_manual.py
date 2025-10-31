from xarray.plot.facetgrid import _nicetitle

# Test cases from the bug report
result = _nicetitle(coord='', value=None, maxchar=1, template='{coord}={value}')
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max length: 1")
print(f"PASS: {len(result) <= 1}")
print()

result2 = _nicetitle(coord='x', value=123, maxchar=2, template='{coord}={value}')
print(f"Result2: '{result2}'")
print(f"Length: {len(result2)}")
print(f"Expected max length: 2")
print(f"PASS: {len(result2) <= 2}")
print()

# Additional test case for maxchar=3 (boundary)
result3 = _nicetitle(coord='test', value='value', maxchar=3, template='{coord}={value}')
print(f"Result3: '{result3}'")
print(f"Length: {len(result3)}")
print(f"Expected max length: 3")
print(f"PASS: {len(result3) <= 3}")
print()

# Test with maxchar=4
result4 = _nicetitle(coord='test', value='value', maxchar=4, template='{coord}={value}')
print(f"Result4: '{result4}'")
print(f"Length: {len(result4)}")
print(f"Expected max length: 4")
print(f"PASS: {len(result4) <= 4}")