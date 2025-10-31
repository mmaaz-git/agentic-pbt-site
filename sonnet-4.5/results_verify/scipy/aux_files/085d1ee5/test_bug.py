from io import StringIO
from scipy.io import arff

# Test case 1: Simple reproduction with trailing newline
print("Test 1: Simple case with trailing newline")
content = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"0.0\n"
"""

try:
    f = StringIO(content)
    data, meta = arff.loadarff(f)
    print("Success! Data loaded without error.")
    print(f"Data length: {len(data)}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 2: Without trailing newline
print("\nTest 2: Simple case without trailing newline")
content2 = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"0.0"
"""

try:
    f = StringIO(content2)
    data, meta = arff.loadarff(f)
    print("Success! Data loaded without error.")
    print(f"Data length: {len(data)}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 3: Multiple rows with trailing newline
print("\nTest 3: Multiple rows with trailing newline")
content3 = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"1.0\n2.0\n3.0\n"
"""

try:
    f = StringIO(content3)
    data, meta = arff.loadarff(f)
    print("Success! Data loaded without error.")
    print(f"Data length: {len(data)}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")

# Test case 4: Multiple rows without trailing newline
print("\nTest 4: Multiple rows without trailing newline")
content4 = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"1.0\n2.0\n3.0"
"""

try:
    f = StringIO(content4)
    data, meta = arff.loadarff(f)
    print("Success! Data loaded without error.")
    print(f"Data length: {len(data)}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")