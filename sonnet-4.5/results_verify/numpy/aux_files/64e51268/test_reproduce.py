import warnings
import numpy.typing as npt

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings emitted: {len(w)}")
    if len(w) > 0:
        for warning in w:
            print(f"  Category: {warning.category}")
            print(f"  Message: {warning.message}")
    else:
        print("  No warnings emitted!")
    assert len(w) == 1, f"Expected 1 DeprecationWarning, got {len(w)}"