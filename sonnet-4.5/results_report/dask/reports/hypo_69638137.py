import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from dask.utils import key_split

@given(st.binary())
def test_key_split_bytes_idempotence(s):
    first_split = key_split(s)
    second_split = key_split(first_split)
    assert first_split == second_split

if __name__ == "__main__":
    test_key_split_bytes_idempotence()