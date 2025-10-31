from Cython.Distutils import Extension

ext = Extension(
    name="test",
    sources=["test.pyx"],
    cython_include_dirs=["path1", "path2"],
    cython_directives={"language_level": 3},
    pyrex_cplus=True,
)

print("cython_include_dirs:", ext.cython_include_dirs)
print("cython_directives:", ext.cython_directives)
print("cython_cplus:", ext.cython_cplus)