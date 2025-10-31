import warnings
from Cython.Build.Dependencies import extended_iglob

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    list(extended_iglob('**/*.py'))
    for warning in w:
        print(f'{warning.category.__name__}: {warning.message}')