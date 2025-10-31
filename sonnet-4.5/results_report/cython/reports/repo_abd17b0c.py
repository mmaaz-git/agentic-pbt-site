import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Cache import Cache
import os
import tempfile
import time

# Create a temporary directory for testing
with tempfile.TemporaryDirectory() as tmpdir:
    # Create a cache with a small size limit (500 bytes)
    cache = Cache(tmpdir, cache_size=500)

    # Create three files with timestamps spaced apart
    # Each file is 200 bytes, total will be 600 bytes (exceeds cache size)

    # Create the oldest file
    old_file = os.path.join(tmpdir, "old.txt")
    with open(old_file, 'w') as f:
        f.write('x' * 200)

    # Wait to ensure different timestamps
    time.sleep(0.1)

    # Create a newer file
    new_file = os.path.join(tmpdir, "new.txt")
    with open(new_file, 'w') as f:
        f.write('x' * 200)

    # Wait to ensure different timestamps
    time.sleep(0.1)

    # Create the newest file
    newest_file = os.path.join(tmpdir, "newest.txt")
    with open(newest_file, 'w') as f:
        f.write('x' * 200)

    # Show files before cleanup
    print(f"Files before cleanup: {sorted(os.listdir(tmpdir))}")

    # Get access times before cleanup
    files_with_times = []
    for fname in os.listdir(tmpdir):
        fpath = os.path.join(tmpdir, fname)
        atime = os.stat(fpath).st_atime
        files_with_times.append((fname, atime))
    files_with_times.sort(key=lambda x: x[1])

    print("\nFiles ordered by access time (oldest to newest):")
    for fname, atime in files_with_times:
        print(f"  {fname}: {atime}")

    # Run cleanup with ratio=0.5 (should keep 250 bytes, so only 1 file)
    cache.cleanup_cache(ratio=0.5)

    # Show files after cleanup
    remaining = set(os.listdir(tmpdir))
    print(f"\nFiles after cleanup: {sorted(remaining)}")

    # Check which files were kept vs removed
    if "old.txt" in remaining and "newest.txt" not in remaining:
        print("\nBUG CONFIRMED: Oldest file kept, newest file removed (opposite of LRU)")
        print("Expected behavior: Keep newest files, remove oldest files")
        print("Actual behavior: Kept oldest file, removed newest files")
    elif "newest.txt" in remaining and "old.txt" not in remaining:
        print("\nCorrect LRU behavior: Newest file kept, oldest file removed")
    else:
        print(f"\nUnexpected result - remaining files: {remaining}")