import os
import sys
import tempfile
import importlib

try:
    import pyximport
    print("pyximport module found")
except ImportError:
    print("pyximport module not found - installing Cython")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "Cython"], check=True)
    import pyximport

# Test 1: Simple reproduction test
print("\n=== Test 1: Simple Reproduction ===")
with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, 'first.py'), 'w') as f:
        f.write('X = 1')
    with open(os.path.join(tmpdir, 'second.py'), 'w') as f:
        f.write('Y = 2')

    sys.path.insert(0, tmpdir)

    try:
        py_imp, pyx_imp = pyximport.install(pyimport=True)

        mod1 = importlib.import_module('first')
        print(f"first module: {mod1.__file__}")
        print(f"first module X value: {mod1.X}")

        mod2 = importlib.import_module('second')
        print(f"second module: {mod2.__file__}")
        print(f"second module Y value: {mod2.Y}")

        # Check if compiled
        is_first_compiled = '.so' in mod1.__file__ or '.pyd' in mod1.__file__
        is_second_compiled = '.so' in mod2.__file__ or '.pyd' in mod2.__file__

        print(f"\nFirst module compiled: {is_first_compiled}")
        print(f"Second module compiled: {is_second_compiled}")

        if is_first_compiled and not is_second_compiled:
            print("\n⚠️ BUG CONFIRMED: First module is compiled but second is not!")
        elif is_first_compiled and is_second_compiled:
            print("\n✓ Both modules are compiled - bug not present")
        else:
            print(f"\n? Unexpected state - neither or different compilation")

    finally:
        sys.path.remove(tmpdir)
        pyximport.uninstall(py_imp, pyx_imp)
        # Clean up imported modules
        if 'first' in sys.modules:
            del sys.modules['first']
        if 'second' in sys.modules:
            del sys.modules['second']