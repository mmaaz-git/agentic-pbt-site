import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import time

# Test with num_files=3 (the failing input from the bug report)
def test_with_3_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir, cache_size=500)

        files = []
        for i in range(3):
            filepath = os.path.join(tmpdir, f"file{i}.txt")
            with open(filepath, 'w') as f:
                f.write('x' * 200)
            time.sleep(0.01)
            os.utime(filepath, None)
            files.append(filepath)

        total_size = sum(os.path.getsize(f) for f in files)
        print(f"Total size: {total_size} bytes, Cache size: {cache.cache_size} bytes")

        if total_size > cache.cache_size:
            print(f"Files before cleanup: {sorted(os.listdir(tmpdir))}")
            cache.cleanup_cache(ratio=0.5)
            remaining = set(os.listdir(tmpdir))
            print(f"Files after cleanup: {sorted(remaining)}")

            oldest_file = os.path.basename(files[0])
            newest_file = os.path.basename(files[-1])

            print(f"Oldest file: {oldest_file}")
            print(f"Newest file: {newest_file}")
            print(f"Is newest file in remaining? {newest_file in remaining}")
            print(f"Is oldest file in remaining? {oldest_file in remaining}")

            assert newest_file in remaining, "Most recently accessed file should be kept"
            print("TEST PASSED")
        else:
            print(f"Total size {total_size} not greater than cache size {cache.cache_size}, test not applicable")

test_with_3_files()