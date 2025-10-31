#!/usr/bin/env python3
"""
Minimal reproduction of scipy.io.arff split_data_line IndexError on empty string.
"""

from scipy.io.arff._arffread import split_data_line

# Direct test with empty string
print("Testing split_data_line with empty string...")
try:
    result, dialect = split_data_line("")
    print(f"Result: {result}, Dialect: {dialect}")
except IndexError as e:
    print(f"IndexError: {e}")

# Also test with loadarff
print("\nTesting loadarff with ARFF containing empty line...")
from scipy.io.arff import loadarff
from io import StringIO

arff_content = """@relation test
@attribute x numeric
@data
1.0

2.0
"""

try:
    data, meta = loadarff(StringIO(arff_content))
    print(f"Successfully loaded ARFF data with {len(data)} records")
except IndexError as e:
    print(f"IndexError during loadarff: {e}")