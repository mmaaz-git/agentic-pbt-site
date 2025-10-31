from scipy.io import arff
from io import StringIO

# Test case 1: Single character attribute name with single quotes
arff1 = "@relation test\n@attribute 'a' numeric\n@data\n1.0"
data1, meta1 = arff.loadarff(StringIO(arff1))
print(f"Single-char with single quotes: '{meta1.names()[0]}'")

# Test case 2: Multi-character attribute name with single quotes
arff2 = "@relation test\n@attribute 'ab' numeric\n@data\n1.0"
data2, meta2 = arff.loadarff(StringIO(arff2))
print(f"Multi-char with single quotes: '{meta2.names()[0]}'")

# Test case 3: Attribute name with double quotes
arff3 = "@relation test\n@attribute \"myattr\" numeric\n@data\n1.0"
data3, meta3 = arff.loadarff(StringIO(arff3))
print(f"With double quotes: '{meta3.names()[0]}'")

# Test case 4: Demonstrate the data access issue
print("\nData access test:")
try:
    print(f"Accessing data3['myattr']: {data3['myattr']}")
except (KeyError, ValueError) as e:
    print(f"Error accessing data3['myattr']: {e}")

try:
    print(f"Accessing data3['\"myattr\"']: {data3['\"myattr\"']}")
except (KeyError, ValueError) as e:
    print(f"Error accessing data3['\"myattr\"']: {e}")