from hypothesis import given, strategies as st, settings
import dask.bag as db
import traceback

@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=20)
def test_variance_division_by_zero(n):
    data = [float(i) for i in range(n)]
    b = db.from_sequence(data, npartitions=1)

    try:
        result = b.var(ddof=n).compute()
        print(f"n={n}: Success, result={result}")
    except ZeroDivisionError as e:
        print(f"n={n}: ZeroDivisionError - {e}")
        return  # Expected error
    except Exception as e:
        print(f"n={n}: Unexpected error - {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_variance_division_by_zero()
    print("\nTest completed!")