import os
import sys
import tempfile
import importlib
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    # Create two simple Python modules
    with open(os.path.join(tmpdir, 'first.py'), 'w') as f:
        f.write('X = 1')
    with open(os.path.join(tmpdir, 'second.py'), 'w') as f:
        f.write('Y = 2')

    # Add tmpdir to path so modules can be imported
    sys.path.insert(0, tmpdir)

    # Install pyximport with pyimport=True to compile .py files
    py_imp, pyx_imp = pyximport.install(pyimport=True)

    # Import first module
    mod1 = importlib.import_module('first')
    print(f"first module file: {mod1.__file__}")
    print(f"first module X value: {mod1.X}")

    # Import second module
    mod2 = importlib.import_module('second')
    print(f"second module file: {mod2.__file__}")
    print(f"second module Y value: {mod2.Y}")

    # Check if modules were compiled (should have .so or .pyd extension)
    compiled_first = '.so' in mod1.__file__ or '.pyd' in mod1.__file__
    compiled_second = '.so' in mod2.__file__ or '.pyd' in mod2.__file__

    print(f"\nfirst module compiled: {compiled_first}")
    print(f"second module compiled: {compiled_second}")

    if compiled_first and not compiled_second:
        print("\nBUG CONFIRMED: Only first module was compiled!")

    # Cleanup
    sys.path.remove(tmpdir)
    pyximport.uninstall(py_imp, pyx_imp)