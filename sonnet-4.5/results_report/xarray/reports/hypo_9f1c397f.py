#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.backends.file_manager import _HashedSequence

@given(
    tuple_value=st.tuples(st.integers(), st.integers()),
)
def test_hashed_sequence_mutation_breaks_hash(tuple_value):
    hashed_seq = _HashedSequence(tuple_value)
    original_hash = hash(hashed_seq)

    hashed_seq.append(999)
    new_hash = hash(hashed_seq)

    assert original_hash == new_hash, \
        "Hash should not change even after mutation (cached hash bug)"

    actual_tuple_hash = hash(tuple(hashed_seq))
    assert new_hash == actual_tuple_hash, \
        "Cached hash should match the hash of current tuple value"

if __name__ == "__main__":
    test_hashed_sequence_mutation_breaks_hash()