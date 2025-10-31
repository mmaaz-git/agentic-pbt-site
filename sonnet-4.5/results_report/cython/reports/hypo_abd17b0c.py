import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Build.Cache import Cache
import os
import tempfile
import time


@given(st.integers(min_value=3, max_value=10))
@settings(max_examples=10)
def test_cache_cleanup_preserves_recently_accessed_files(num_files):
    """Test that cache cleanup removes oldest files first (LRU policy)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache with small size to force cleanup
        cache = Cache(tmpdir, cache_size=500)

        # Create files with increasing access times
        files = []
        for i in range(num_files):
            filepath = os.path.join(tmpdir, f"file{i}.txt")
            with open(filepath, 'w') as f:
                f.write('x' * 200)  # Each file is 200 bytes
            time.sleep(0.01)  # Ensure different timestamps
            os.utime(filepath, None)  # Update access time
            files.append(filepath)

        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in files)

        # Only test cleanup if we exceed cache size
        if total_size > cache.cache_size:
            # Run cleanup with ratio=0.5 (keep 50% of cache size)
            cache.cleanup_cache(ratio=0.5)

            # Check what remains
            remaining = set(os.listdir(tmpdir))

            # Get the oldest and newest file names
            oldest_file = os.path.basename(files[0])
            newest_file = os.path.basename(files[-1])

            # The newest file should be kept (LRU policy)
            assert newest_file in remaining, f"Most recently accessed file '{newest_file}' should be kept, but was removed. Remaining files: {remaining}"


# Run the test
if __name__ == "__main__":
    test_cache_cleanup_preserves_recently_accessed_files()