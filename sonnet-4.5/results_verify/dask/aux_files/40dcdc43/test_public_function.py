#!/usr/bin/env python3
"""Test to see how the public read_orc function handles the columns parameter"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

# Look at how the public read_orc function handles columns before passing to _read_orc
def examine_public_function_behavior():
    print("Looking at public read_orc function (lines 97-108):")
    print("""
    if columns is not None and index in columns:
        columns = [col for col in columns if col != index]
    return dd.from_map(
        _read_orc,
        parts,
        engine=engine,
        fs=fs,
        schema=schema,
        index=index,
        meta=meta,
        columns=columns,
    )
    """)

    print("\nKey observation:")
    print("The public function creates a NEW list when it needs to modify columns:")
    print("    columns = [col for col in columns if col != index]")
    print("This creates a new list rather than mutating the original.")
    print("\nHowever, _read_orc then mutates this list by appending to it:")
    print("    columns.append(index)")

if __name__ == "__main__":
    examine_public_function_behavior()

    print("\n" + "="*60)
    print("Analysis of the interaction between public and private functions:")
    print("="*60)
    print("""
1. Public read_orc function (line 97-98):
   - If index is in columns, it creates a NEW list without the index
   - This is good practice - it doesn't mutate the input

2. Private _read_orc function (line 112-113):
   - If index is not None and columns is not None, it MUTATES the list
   - It appends the index back to the columns list

3. The problem:
   - When read_orc creates a new list and passes it to _read_orc,
     the mutation doesn't affect the original user's list (lucky!)
   - But when dd.from_map calls _read_orc multiple times with the
     same columns list, the mutation persists across calls
   - Also, if someone calls _read_orc directly (even though it's private),
     they'll experience unexpected mutation

4. Why this matters:
   - dd.from_map might call _read_orc multiple times with the same
     columns reference for different partitions
   - Each call would keep appending the index, growing the list
""")