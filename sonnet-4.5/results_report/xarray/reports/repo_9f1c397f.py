#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.backends.file_manager import _HashedSequence

# Create a _HashedSequence with an initial tuple
original_tuple = (1, 2, 3)
hashed_seq = _HashedSequence(original_tuple)

print(f"Original: {list(hashed_seq)}, hash={hash(hashed_seq)}")

# Mutate the _HashedSequence by appending an element
hashed_seq.append(999)

print(f"After mutation: {list(hashed_seq)}, hash={hash(hashed_seq)}")
print(f"Expected hash: {hash(tuple(hashed_seq))}")

# Demonstrate the bug: hash is stale after mutation
if hash(hashed_seq) != hash(tuple(hashed_seq)):
    print("Bug: hash is stale after mutation")
    print(f"Cached hash: {hash(hashed_seq)}")
    print(f"Correct hash for current content: {hash(tuple(hashed_seq))}")
else:
    print("No bug detected")