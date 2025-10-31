from scipy.io.arff._arffread import NominalAttribute

print("Test 1: Direct NominalAttribute instantiation with empty values")
try:
    attr = NominalAttribute("test_attr", ())
    print("Success: Created NominalAttribute with empty values")
except ValueError as e:
    print(f"Crash: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "="*50 + "\n")

print("Test 2: Trying with empty list instead of tuple")
try:
    attr = NominalAttribute("test_attr", [])
    print("Success: Created NominalAttribute with empty list")
except ValueError as e:
    print(f"Crash: {e}")
    print(f"Error type: {type(e).__name__}")

print("\n" + "="*50 + "\n")

from io import StringIO
from scipy.io.arff import loadarff

print("Test 3: Loading ARFF file with empty nominal attribute")
arff_content = """@RELATION test
@ATTRIBUTE color {}
@DATA
"""

try:
    data, meta = loadarff(StringIO(arff_content))
    print("Success: Loaded ARFF with empty nominal attribute")
except Exception as e:
    print(f"Crash when loading ARFF: {e}")
    print(f"Error type: {type(e).__name__}")