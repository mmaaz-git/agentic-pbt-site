import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase

    if len(w) == 0:
        print("BUG: No deprecation warning was emitted when accessing NBitBase")
    else:
        print(f"OK: {len(w)} warning(s) emitted")
        for warning in w:
            print(f"  - {warning.category.__name__}: {warning.message}")