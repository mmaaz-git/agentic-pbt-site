from hypothesis import given, strategies as st
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt

@given(st.integers(min_value=0, max_value=10))
def test_parallel_coordinates_handles_no_numeric_cols(n_rows):
    """Test from the bug report"""
    df = pd.DataFrame({'class': ['a'] * n_rows})
    fig, ax = plt.subplots()
    try:
        result = pandas.plotting.parallel_coordinates(df, 'class')
        assert result is not None
        print(f"n_rows={n_rows}: SUCCESS (unexpected - function should fail)")
    except ValueError as e:
        # This is what the test expects - a ValueError with "numeric" or "column"
        assert "numeric" in str(e).lower() or "column" in str(e).lower()
        print(f"n_rows={n_rows}: Got expected ValueError: {e}")
    except IndexError as e:
        # This is what actually happens
        print(f"n_rows={n_rows}: Got IndexError (BUG): {e}")
    finally:
        plt.close(fig)

# Run the test
print("Running hypothesis test...")
test_parallel_coordinates_handles_no_numeric_cols()
print("\nAll test cases passed (but with IndexError instead of ValueError)")