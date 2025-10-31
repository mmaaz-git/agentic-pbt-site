import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# Create a simple extension
ext = Extension("test_module", ["test.pyx"])

setup(
    name="TestPackage",
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext}
)