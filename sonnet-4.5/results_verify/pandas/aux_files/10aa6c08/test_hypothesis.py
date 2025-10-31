import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Compiler.Main import _make_range_re

@given(st.text())
def test_make_range_re_handles_all_lengths(chrs):
    result = _make_range_re(chrs)
    
test_make_range_re_handles_all_lengths()
