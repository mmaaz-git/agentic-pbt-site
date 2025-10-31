import sys
import builtins

# Save the original import function
original_import = builtins.__import__

def custom_import(name, *args, **kwargs):
    if name == 'pooch':
        raise ImportError("No module named 'pooch'")
    return original_import(name, *args, **kwargs)

# Replace the import function
builtins.__import__ = custom_import

try:
    # Now try to use the _download_all module from scipy
    sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/')

    # First delete any cached scipy modules
    for key in list(sys.modules.keys()):
        if 'scipy' in key or 'pooch' in key:
            del sys.modules[key]

    # Now import fresh
    from scipy.datasets import _download_all

    # Try calling main directly as a script would
    _download_all.main()

except AttributeError as e:
    print(f"Got AttributeError: {e}")
    print("This is the bug - it should raise ImportError with a clear message")
except ImportError as e:
    print(f"Got ImportError: {e}")
    print("This would be the expected behavior")
finally:
    # Restore the original import
    builtins.__import__ = original_import