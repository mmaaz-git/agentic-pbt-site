from io import StringIO
from scipy.io import arff
import traceback

print("Testing direct reproduction from the bug report...")

arff_content = """@relation test
@attribute width numeric
@attribute width numeric
@data
5.0,3.25
4.5,3.75
"""

print(f"ARFF content:\n{arff_content}")

f = StringIO(arff_content)
try:
    data, meta = arff.loadarff(f)
    print("Loaded successfully (unexpected!)")
    print(f"Data: {data}")
    print(f"Meta: {meta}")
except Exception as e:
    print(f"\nException type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()