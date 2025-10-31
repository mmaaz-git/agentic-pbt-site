from hypothesis import given, strategies as st, settings
import pandas as pd
import io

@given(
    unhashable_names=st.lists(
        st.one_of(
            st.lists(st.integers(), min_size=1, max_size=3),
            st.dictionaries(st.text(min_size=1, max_size=3), st.integers(), min_size=1, max_size=2)
        ),
        min_size=2,
        max_size=5
    )
)
@settings(max_examples=50)
def test_unhashable_names_should_raise_valueerror(unhashable_names):
    csv_data = ','.join(['0'] * len(unhashable_names)) + '\n'
    try:
        df = pd.read_csv(io.StringIO(csv_data), names=unhashable_names, header=None)
        assert False, "Should have raised an error"
    except TypeError:
        assert False, "Got TypeError but docstring promises ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_unhashable_names_should_raise_valueerror()