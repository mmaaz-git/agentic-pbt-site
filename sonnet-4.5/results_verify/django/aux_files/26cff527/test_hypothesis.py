#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings, example
from django.core.cache.utils import make_template_fragment_key

# Run the exact test from the bug report
@given(
    fragment_name=st.text(),
    list1=st.lists(st.text(), min_size=1, max_size=5),
    list2=st.lists(st.text(), min_size=1, max_size=5)
)
@example(fragment_name="test", list1=["a:", "b"], list2=["a", ":b"])
@example(fragment_name="test", list1=["x:y", "z"], list2=["x", "y:z"])
@settings(max_examples=100)
def test_different_inputs_should_produce_different_keys(fragment_name, list1, list2):
    assume(list1 != list2)

    key1 = make_template_fragment_key(fragment_name, list1)
    key2 = make_template_fragment_key(fragment_name, list2)

    if key1 == key2:
        print(f"\nFound collision!")
        print(f"Fragment name: {fragment_name!r}")
        print(f"List 1: {list1}")
        print(f"List 2: {list2}")
        print(f"Both produce key: {key1}")

        # Show what's happening
        import hashlib
        hasher = hashlib.md5(usedforsecurity=False)
        for arg in list1:
            hasher.update(str(arg).encode())
            hasher.update(b":")
        hash1 = hasher.hexdigest()

        hasher = hashlib.md5(usedforsecurity=False)
        for arg in list2:
            hasher.update(str(arg).encode())
            hasher.update(b":")
        hash2 = hasher.hexdigest()

        print(f"Hash portion: {hash1}")
        assert False, f"Cache key collision: {list1} and {list2} produce same key"

print("Running hypothesis test...")
try:
    test_different_inputs_should_produce_different_keys()
    print("Test passed - no collisions found")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")