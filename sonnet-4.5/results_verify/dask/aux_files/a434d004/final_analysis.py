#!/usr/bin/env python3
"""Final analysis of the LRU bug"""

from collections import OrderedDict, UserDict
from typing import TypeVar, cast
from collections.abc import Hashable
from dask.dataframe.dask_expr._util import LRU

def analyze_bug_report_claim():
    """Carefully analyze what the bug report is claiming"""
    print("=" * 60)
    print("Analyzing the bug report's specific claim")
    print("=" * 60)

    print("\nThe bug report test case with maxsize=1:")
    print("1. Create LRU with maxsize=1")
    print("2. Add key 0 with value 'value0'")
    print("3. ACCESS key 0 (calls __getitem__)")
    print("4. Add key 1 with value 'value1'")
    print("5. Assert that key 0 is still in cache")

    print("\nThe expectation seems to be that:")
    print("- Accessing key 0 marks it as 'recently used'")
    print("- Therefore it shouldn't be evicted")

    print("\nBut wait - with maxsize=1, only ONE item can exist!")
    print("If we add key 1, key 0 MUST be evicted (there's no room for both)")

    print("\nLet me re-read the docstring...")
    lru = LRU(1)
    print(f"\nDocstring: '{LRU.__doc__}'")

    print("\nIt says: 'Limited size mapping, evicting the least recently")
    print("looked-up key when full'")

    print("\nSo the question is: Is key 0 the 'least recently looked-up'")
    print("after we just accessed it? NO! It's the MOST recently looked-up!")

    print("\nBut with size=1, it's also the ONLY item, so it's both")
    print("the most AND least recently used.")

    print("\nAH! I think I understand the confusion now!")
    print("\nLet me test a specific scenario...")

def test_the_real_issue():
    """Test what might be the real issue"""
    print("\n" + "=" * 60)
    print("Testing potential real issue")
    print("=" * 60)

    print("\nMaybe the issue is about WHEN the eviction happens")
    print("during __setitem__. Let's trace through it:")

    print("\nCurrent buggy implementation:")
    lru = LRU(maxsize=1)
    lru[0] = "value0"
    print(f"1. After lru[0] = 'value0': {dict(lru)}")

    # Now let's see what happens when we UPDATE the same key
    print("\n2. Now calling lru[0] = 'updated_value0' (UPDATE existing key)")
    print("   Current __setitem__ code:")
    print("   if len(self) >= self.maxsize:  # 1 >= 1 is True")
    print("       popitem(last=False)  # This removes key 0!")
    print("   super().__setitem__(0, 'updated_value0')  # Then adds it back")

    lru[0] = "updated_value0"
    print(f"   Result: {dict(lru)}")

    print("\nSo the bug is that when UPDATING an existing key,")
    print("it unnecessarily evicts and re-adds the key!")

    print("\nThis is clearly wrong because:")
    print("1. Updating a key shouldn't trigger eviction")
    print("2. The cache size doesn't change when updating")

def test_with_proper_lru_semantics():
    """Test what proper LRU semantics should be"""
    print("\n" + "=" * 60)
    print("What SHOULD happen (proper LRU semantics)")
    print("=" * 60)

    print("\nScenario 1: maxsize=2, demonstrate proper LRU")
    print("-" * 40)

    # Using current buggy implementation
    lru = LRU(maxsize=2)
    lru[0] = "v0"
    lru[1] = "v1"
    print(f"Initial: {dict(lru)}")

    _ = lru[0]  # Access 0
    print(f"After accessing key 0: {dict(lru)}")
    print("(OrderedDict moved key 0 to end = most recently used)")

    lru[2] = "v2"
    print(f"After adding key 2: {dict(lru)}")
    print(f"Key 1 evicted? {1 not in lru}, Key 0 kept? {0 in lru}")

    print("\nThis works correctly! Key 1 was evicted (LRU), key 0 kept (MRU)")

    print("\nScenario 2: maxsize=1 (the contentious case)")
    print("-" * 40)

    lru = LRU(maxsize=1)
    lru[0] = "v0"
    print(f"Initial: {dict(lru)}")

    _ = lru[0]  # Access 0
    print(f"After accessing key 0: {dict(lru)}")

    lru[1] = "v1"
    print(f"After adding key 1: {dict(lru)}")

    print("\nIs this correct behavior?")
    print("- Cache size is 1, so only 1 item can exist")
    print("- We're adding a NEW item (key 1)")
    print("- The old item (key 0) must be evicted")
    print("- The fact that we accessed key 0 doesn't change this")
    print("- There's simply no room for both!")

    print("\nConclusion: For maxsize=1, this might NOT be a bug!")
    print("When adding a NEW item to a full size-1 cache,")
    print("the existing item MUST be evicted regardless of access pattern.")

if __name__ == "__main__":
    analyze_bug_report_claim()
    test_the_real_issue()
    test_with_proper_lru_semantics()