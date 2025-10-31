import sys
from pandas.compat._optional import import_optional_dependency


parent_module_name = "fake_parent_xyz"
submodule_name = f"{parent_module_name}.submodule"

class FakeSubmodule:
    __name__ = submodule_name
    __version__ = "1.0.0"

sys.modules[submodule_name] = FakeSubmodule()

if parent_module_name in sys.modules:
    del sys.modules[parent_module_name]

try:
    import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
except ImportError as e:
    print(f"ImportError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
finally:
    del sys.modules[submodule_name]