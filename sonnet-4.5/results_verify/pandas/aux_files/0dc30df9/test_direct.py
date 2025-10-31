import pandas.core.computation.parsing as parsing
import traceback

name = '\x00'
print(f"Testing with input: {repr(name)}")

try:
    result = parsing.clean_column_name(name)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
    traceback.print_exc()