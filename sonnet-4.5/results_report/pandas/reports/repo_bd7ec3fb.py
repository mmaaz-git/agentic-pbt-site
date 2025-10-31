from pandas.io.json._normalize import nested_to_record

# Test case that crashes
d = {1: {2: 'value'}}
try:
    result = nested_to_record(d)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print()

# More complex example
d2 = {1: 'a', 2: 'b', 3: {4: 'c', 5: 'd'}}
try:
    result2 = nested_to_record(d2)
    print(f"Result: {result2}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")