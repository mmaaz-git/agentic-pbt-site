import os
import tempfile
import zipfile

def simulate_actual_cython_behavior():
    """
    Simulate exactly what happens in Cython's Cache.load_from_cache
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up paths as they would be in actual use
        work_dir = os.path.join(tmpdir, "work")
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(work_dir)
        os.makedirs(cache_dir)

        # The c_file path where we expect files to be extracted
        c_file = os.path.join(work_dir, "module.c")

        # Create a cache zip with multiple artifacts (as store_to_cache does)
        cache_zip = os.path.join(cache_dir, "module.c-fingerprint.zip")
        with zipfile.ZipFile(cache_zip, 'w') as z:
            # Note: store_to_cache uses os.path.basename, so no paths in zip
            z.writestr("module.c", "/* C source */")
            z.writestr("module.h", "/* C header */")
            z.writestr("module_api.h", "/* API header */")

        print("Cache zip contents:")
        with zipfile.ZipFile(cache_zip, 'r') as z:
            print(z.namelist())

        # Now simulate load_from_cache
        print(f"\nSimulating load_from_cache with c_file = {c_file}")
        dirname = os.path.dirname(c_file)
        print(f"dirname = {dirname}")

        with zipfile.ZipFile(cache_zip, 'r') as z:
            for artifact in z.namelist():
                # This is the BUGGY line 155 in Cache.py:
                buggy_path = os.path.join(dirname, artifact)
                print(f"\nExtracting '{artifact}':")
                print(f"  Buggy: z.extract('{artifact}', '{buggy_path}')")
                print(f"  Fixed: z.extract('{artifact}', '{dirname}')")

                # Actually do the buggy extraction
                z.extract(artifact, buggy_path)

        print("\n\nResult of buggy extraction:")
        print("Expected files at:")
        for artifact in ["module.c", "module.h", "module_api.h"]:
            expected_path = os.path.join(work_dir, artifact)
            print(f"  {expected_path}: exists={os.path.exists(expected_path)}, is_file={os.path.isfile(expected_path)}")

        print("\nActual directory structure:")
        for root, dirs, files in os.walk(work_dir):
            level = root.replace(work_dir, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = '  ' * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

        # Now test if this would break compilation
        print("\n\nWould this break compilation?")
        print("-" * 50)

        # Check if we can read the "extracted" files
        for artifact in ["module.c", "module.h", "module_api.h"]:
            expected_path = os.path.join(work_dir, artifact)
            if os.path.isfile(expected_path):
                with open(expected_path, 'r') as f:
                    content = f.read()
                print(f"✓ Can read {artifact}: {repr(content[:20])}")
            elif os.path.isdir(expected_path):
                print(f"✗ {artifact} is a directory, not a file!")
                actual_file = os.path.join(expected_path, artifact)
                if os.path.isfile(actual_file):
                    print(f"  Actual file is at: {actual_file}")
            else:
                print(f"✗ {artifact} doesn't exist at expected location")

if __name__ == "__main__":
    simulate_actual_cython_behavior()