import scipy.odr
import pickle

print("Testing scipy.odr.polynomial pickling...")
try:
    poly_model = scipy.odr.polynomial(2)
    pickled = pickle.dumps(poly_model)
    print("SUCCESS: polynomial(2) can be pickled")
except Exception as e:
    print(f"FAILED: polynomial(2) cannot be pickled")
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting other ODR models...")
models = [
    ("multilinear", scipy.odr.multilinear),
    ("exponential", scipy.odr.exponential),
    ("quadratic", scipy.odr.quadratic),
    ("unilinear", scipy.odr.unilinear)
]

for name, model in models:
    try:
        pickled = pickle.dumps(model)
        unpickled = pickle.loads(pickled)
        print(f"SUCCESS: {name} can be pickled and unpickled")
    except Exception as e:
        print(f"FAILED: {name} cannot be pickled")
        print(f"Error: {type(e).__name__}: {e}")

print("\nTesting the Hypothesis test case...")
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    n_points=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=10000)
)
@settings(max_examples=5)
def test_pickling_round_trip(n_points, seed):
    poly_model = scipy.odr.polynomial(2)
    model_pickled = pickle.loads(pickle.dumps(poly_model))
    assert model_pickled is not None
    print(f"  Test passed with n_points={n_points}, seed={seed}")

try:
    test_pickling_round_trip()
    print("Hypothesis test passed all examples")
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}: {e}")