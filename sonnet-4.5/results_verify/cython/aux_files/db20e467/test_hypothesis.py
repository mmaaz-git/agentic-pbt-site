import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import os
from hypothesis import given, strategies as st, settings
from Cython.Debugger.Cygdb import make_command_file


@settings(max_examples=100)
@given(st.text(alphabet=st.characters(whitelist_categories=('Cs',)), min_size=1, max_size=10))
def test_surrogate_characters_cause_crash(prefix_code):
    result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)
    try:
        with open(result, 'r') as f:
            f.read()
    finally:
        if os.path.exists(result):
            os.remove(result)

if __name__ == "__main__":
    test_surrogate_characters_cause_crash()