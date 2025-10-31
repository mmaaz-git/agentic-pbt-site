from hypothesis import given, strategies as st


def complete_unqualified_logic(text, word, all_names):
    word = word or ""
    seen = set(text[:-len(word)].split())
    return [n for n in all_names if n.startswith(word) and n not in seen]


@given(st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
def test_complete_with_empty_word(funcname):
    word = ""
    text = f"cy break {funcname} "
    all_names = [funcname, "other_func", "another_func"]

    result = complete_unqualified_logic(text, word, all_names)

    assert funcname in result


# Run specific failing case
print("Testing with funcname='spam', word='':")
funcname = "spam"
word = ""
text = f"cy break {funcname} "
all_names = [funcname, "other_func", "another_func"]

result = complete_unqualified_logic(text, word, all_names)
print(f"Result: {result}")
print(f"funcname in result: {funcname in result}")

# Run hypothesis test
test_complete_with_empty_word()