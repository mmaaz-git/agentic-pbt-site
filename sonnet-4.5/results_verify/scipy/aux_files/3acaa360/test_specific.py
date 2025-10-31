import numpy as np
import scipy.linalg
import warnings

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    A = np.array([[0., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])

    print("Matrix A:")
    print(A)
    print(f"\nA squared:")
    print(A @ A)
    print(f"Is A nilpotent (A^2 = 0)? {np.allclose(A @ A, 0)}")

    funm_result = scipy.linalg.funm(A, np.exp)
    expm_result = scipy.linalg.expm(A)

    print("\nfunm(A, exp) =")
    print(funm_result)

    print("\nexpm(A) =")
    print(expm_result)

    print("\nExpected (I + A) =")
    print(np.eye(3) + A)

    print("\nAre funm and expm results equal?", np.allclose(funm_result, expm_result))
    print("Are funm and expm results equal (strict)?", np.array_equal(funm_result, expm_result))

    print("\nDifference between funm and expm:")
    print(expm_result - funm_result)

    print(f"\nWarnings captured: {len(w)}")
    for warning in w:
        print(f"  - {warning.message}")

    # Verify the assertion from the bug report
    try:
        assert not np.allclose(funm_result, expm_result), \
            f"funm returns identity, but expm correctly returns I + A"
        print("\nAssertion passed: funm and expm give different results")
    except AssertionError as e:
        print(f"\nAssertion failed: {e}")