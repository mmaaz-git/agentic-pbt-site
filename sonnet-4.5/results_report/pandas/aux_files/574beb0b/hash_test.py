import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas._libs.hashtable as htable
import numpy as np

# Test string hash table directly
print("Testing StringHashTable:")
arr = np.array(['', '\x00'], dtype=object)
table = htable.StringHashTable()
uniques = table.unique(arr)
print(f"Input:   {repr(list(arr))}")
print(f"Uniques: {repr(list(uniques))}")
print()

print("Testing PyObjectHashTable:")
table2 = htable.PyObjectHashTable()
uniques2 = table2.unique(arr)
print(f"Input:   {repr(list(arr))}")
print(f"Uniques: {repr(list(uniques2))}")
print()

# Test the factorize method directly
print("Testing StringHashTable.factorize:")
table3 = htable.StringHashTable()
codes, uniques3 = table3.factorize(arr, na_sentinel=-1, na_value=None)
print(f"Input:   {repr(list(arr))}")
print(f"Codes:   {codes}")
print(f"Uniques: {repr(list(uniques3))}")