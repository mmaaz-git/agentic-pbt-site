from scipy.io import arff
from io import StringIO

print("Testing quote stripping behavior:")
print("-" * 40)

# Test 1: Single quotes with single character
arff1 = "@relation test\n@attribute 'a' numeric\n@data\n1.0"
data1, meta1 = arff.loadarff(StringIO(arff1))
print(f"Single quotes, single char 'a': {repr(meta1.names()[0])}")

# Test 2: Single quotes with multiple characters
arff2 = "@relation test\n@attribute 'ab' numeric\n@data\n1.0"
data2, meta2 = arff.loadarff(StringIO(arff2))
print(f"Single quotes, multi char 'ab': {repr(meta2.names()[0])}")

# Test 3: Double quotes
arff3 = "@relation test\n@attribute \"myattr\" numeric\n@data\n1.0"
data3, meta3 = arff.loadarff(StringIO(arff3))
print(f"Double quotes \"myattr\": {repr(meta3.names()[0])}")

print("\n" + "-" * 40)
print("Expected output:")
print("Single quotes, single char 'a': 'a'")
print("Single quotes, multi char 'ab': 'ab'")
print("Double quotes \"myattr\": 'myattr'")

print("\nActual output shows:")
print("- Single char with single quotes: quotes NOT stripped")
print("- Multi char with single quotes: quotes stripped")
print("- Double quotes: quotes NOT stripped")

# Test data access
print("\n" + "-" * 40)
print("Testing data access:")
try:
    val = data3["myattr"]
    print("data3['myattr'] works: ", val)
except Exception as e:
    print(f"data3['myattr'] failed: {e}")

try:
    val = data3['"myattr"']
    print('data3["\\"myattr\\""] works: ', val)
except Exception as e:
    print(f'data3["\\"myattr\\""] failed: {e}')