import pandas.io.common as pd_common
import sys

# Demonstrate the bug: calling dedup_names with duplicate strings
# and is_potential_multiindex=True causes an AssertionError
names = ['0', '0']
print(f"Input names: {names}")
print(f"is_potential_multiindex: True")
print("\nAttempting to call dedup_names...")
sys.stdout.flush()

result = pd_common.dedup_names(names, is_potential_multiindex=True)
print(f"Result: {result}")