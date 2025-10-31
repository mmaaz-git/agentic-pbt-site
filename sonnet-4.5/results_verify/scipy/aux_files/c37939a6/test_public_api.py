#!/usr/bin/env python3
"""Test if DateAttribute is reachable through public API"""

from scipy.io.arff import loadarff
import tempfile
import os
import traceback

# Test 1: Try loading ARFF with date attribute
print("Test 1: Loading ARFF with date attribute")
print("=" * 40)

content1 = """@relation test
@attribute mydate date 'yyyy-MM-dd'
@attribute value numeric

@data
'2023-01-15', 1.0
'2023-02-20', 2.0
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.arff', delete=False) as f:
    f.write(content1)
    fname1 = f.name

try:
    data, meta = loadarff(fname1)
    print(f"Success! Loaded data with shape: {data.shape}")
    print(f"Meta attributes: {meta.names()}")
    print(f"Date attribute type: {meta['mydate']}")
except NotImplementedError as e:
    print(f"NotImplementedError as documented: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    os.unlink(fname1)

# Test 2: Try with empty date format to trigger our bug
print("\n\nTest 2: Loading ARFF with empty date format")
print("=" * 40)

content2 = """@relation test
@attribute mydate date ''
@attribute value numeric

@data
'', 1.0
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.arff', delete=False) as f:
    f.write(content2)
    fname2 = f.name

try:
    data, meta = loadarff(fname2)
    print(f"Success! Loaded data with shape: {data.shape}")
    print(f"Meta attributes: {meta.names()}")
    print(f"Date attribute: {meta['mydate']}")
except ValueError as e:
    print(f"ValueError (expected if bug was fixed): {e}")
except NotImplementedError as e:
    print(f"NotImplementedError as documented: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    os.unlink(fname2)

# Test 3: Check if we can directly access DateAttribute
print("\n\nTest 3: Direct DateAttribute access")
print("=" * 40)

try:
    from scipy.io.arff._arffread import DateAttribute
    print("DateAttribute class is accessible from _arffread module")

    # This is the bug scenario
    attr = DateAttribute.parse_attribute('test', "date ''")
    print(f"BUG CONFIRMED: Empty format didn't raise error")
    print(f"Result: datetime_unit={attr.datetime_unit}, date_format='{attr.date_format}'")
except ImportError:
    print("DateAttribute not accessible")
except ValueError as e:
    print(f"ValueError raised (bug fixed): {e}")