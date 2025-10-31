import sys
from pandas.compat._optional import import_optional_dependency

# Set up the test scenario
parent_module_name = "fake_parent_xyz"
submodule_name = f"{parent_module_name}.submodule"

# Create a fake submodule with a version
class FakeSubmodule:
    __name__ = submodule_name
    __version__ = "1.0.0"

# Add the submodule to sys.modules
sys.modules[submodule_name] = FakeSubmodule()

# Ensure the parent module is NOT in sys.modules
if parent_module_name in sys.modules:
    del sys.modules[parent_module_name]

# Try to import the submodule with a min_version requirement
try:
    result = import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")
    print(f"Successfully imported: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
except ImportError as e:
    print(f"ImportError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
finally:
    # Clean up sys.modules
    if submodule_name in sys.modules:
        del sys.modules[submodule_name]