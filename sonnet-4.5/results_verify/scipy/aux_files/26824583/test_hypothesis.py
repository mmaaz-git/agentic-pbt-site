from hypothesis import given, strategies as st, settings
from scipy.io import arff
from io import StringIO
import traceback


@st.composite
def valid_arff_identifier(draw):
    first_char = draw(st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_', max_size=20))
    return first_char + rest


@st.composite
def arff_mixed_attributes(draw):
    num_attrs = draw(st.integers(min_value=1, max_value=8))
    attributes = []
    for i in range(num_attrs):
        attr_name = draw(valid_arff_identifier())
        is_numeric = draw(st.booleans())
        if is_numeric:
            attributes.append((attr_name, 'numeric'))
        else:
            num_values = draw(st.integers(min_value=2, max_value=5))
            nominal_values = [draw(valid_arff_identifier()) for _ in range(num_values)]
            attributes.append((attr_name, tuple(set(nominal_values))))
    return attributes


def generate_arff_content(relation_name, attributes, data_rows):
    lines = [f"@relation {relation_name}"]
    for attr_name, attr_type in attributes:
        if isinstance(attr_type, str):
            lines.append(f"@attribute {attr_name} {attr_type}")
        else:
            nominal_values = ','.join(attr_type)
            lines.append(f"@attribute {attr_name} {{{nominal_values}}}")
    lines.append("@data")
    lines.extend(data_rows)
    return '\n'.join(lines)


@settings(max_examples=200)
@given(
    relation_name=valid_arff_identifier(),
    attributes=arff_mixed_attributes(),
    data_rows=st.lists(st.text(), min_size=1, max_size=20)
)
def test_loadarff_handles_all_valid_inputs(relation_name, attributes, data_rows):
    content = generate_arff_content(relation_name, attributes, data_rows)
    f = StringIO(content)
    try:
        data, meta = arff.loadarff(f)
    except Exception as e:
        print(f"Failed on input: relation_name={relation_name}, attributes={attributes}, data_rows={data_rows[:3]}")
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"Content:\n{content[:500]}")
        raise

# Test the specific failing input mentioned in the report
def test_specific_failing_input():
    print("Testing the specific failing input from the bug report...")
    relation_name = 'a'
    attributes = [('a', 'numeric'), ('a', 'numeric')]
    data_rows = ['0.0,0.0']

    content = generate_arff_content(relation_name, attributes, data_rows)
    print(f"Generated ARFF content:\n{content}")

    f = StringIO(content)
    try:
        data, meta = arff.loadarff(f)
        print("Successfully loaded (unexpected!)")
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print(f"Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # First test the specific failing input
    test_specific_failing_input()
    print("\n" + "="*50 + "\n")

    # Then run the hypothesis test
    print("Running hypothesis tests...")
    try:
        test_loadarff_handles_all_valid_inputs()
        print("All hypothesis tests passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")