import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
from hypothesis import given, strategies as st, settings, example

@given(
    st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10),
    st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10)
)
@settings(max_examples=300)
@example('abc', 'abd')
def test_commonprefix_not_valid_path(prefix1, prefix2):
    path1 = f"/tmp/very_long_directory_name_{prefix1}/subdir/file.pyx"
    path2 = f"/tmp/very_long_directory_name_{prefix2}/build"

    common = os.path.commonprefix([path1, path2])

    if len(common) > 30 and common != '/':
        assert os.path.isdir(common), \
            f"commonprefix returned invalid directory: {common}"

if __name__ == "__main__":
    test_commonprefix_not_valid_path()