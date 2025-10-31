import numpy.rec as rec

records = [(1, '\x00', 2.0)]
result = rec.fromrecords(records, names=['a', 'b', 'c'])

print("Input:", repr(records[0][1]))
print("Output:", repr(result[0].b))
print("Lengths:", len(records[0][1]), "vs", len(result[0].b))
print("Types:", type(records[0][1]), "vs", type(result[0].b))

# Test assertion
try:
    assert result[0].b == '\x00'
    print("Test PASSED")
except AssertionError as e:
    print("Test FAILED: AssertionError")

# Also test with the workaround mentioned
print("\n--- Testing workaround with bytes ---")
records_bytes = [(1, b'\x00', 2.0)]
result_bytes = rec.fromrecords(records_bytes, names=['a', 'b', 'c'], formats=['i4', 'S1', 'f8'])
print("Bytes input:", repr(records_bytes[0][1]))
print("Bytes output:", repr(result_bytes[0].b))
print("Do they match?", result_bytes[0].b == b'\x00')