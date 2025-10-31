import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Distutils import Extension

ext = Extension(
    "mymodule",
    ["mymodule.pyx"],
    pyrex_gdb=True,
    cython_include_dirs=["/usr/local/include"],
    cython_directives={"boundscheck": False}
)

print(f"cython_include_dirs: {ext.cython_include_dirs}")
print(f"cython_directives: {ext.cython_directives}")
print(f"cython_gdb: {ext.cython_gdb}")