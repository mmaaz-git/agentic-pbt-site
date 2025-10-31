import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Cache import Cache
import os
import tempfile
import time


@given(st.integers(min_value=3, max_value=10))
def test_cache_cleanup_preserves_recently_accessed_files(num_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir, cache_size=500)

        files = []
        for i in range(num_files):
            filepath = os.path.join(tmpdir, f"file{i}.txt")
            with open(filepath, 'w') as f:
                f.write('x' * 200)
            time.sleep(0.01)
            os.utime(filepath, None)
            files.append(filepath)

        total_size = sum(os.path.getsize(f) for f in files)

        if total_size > cache.cache_size:
            cache.cleanup_cache(ratio=0.5)
            remaining = set(os.listdir(tmpdir))

            oldest_file = os.path.basename(files[0])
            newest_file = os.path.basename(files[-1])

            assert newest_file in remaining, "Most recently accessed file should be kept"

if __name__ == "__main__":
    # Run with num_files=3 to test the failing input
    test_cache_cleanup_preserves_recently_accessed_files(3)
    print("Test completed")