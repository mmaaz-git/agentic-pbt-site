import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from numpy.f2py.symbolic import as_symbol, as_integer, Expr, Op

@st.composite
def expr_integers(draw):
    value = draw(st.integers(min_value=-1000, max_value=1000))
    kind = draw(st.sampled_from([4, 8]))
    return as_integer(value, kind)

@st.composite
def expr_symbols(draw):
    name = draw(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
    return as_symbol(name)

@st.composite
def expr_trees(draw, max_depth=3):
    if max_depth == 0:
        choice = draw(st.sampled_from(['int', 'symbol']))
        if choice == 'int':
            return draw(expr_integers())
        else:
            return draw(expr_symbols())

    choice = draw(st.sampled_from(['simple', 'add']))
    if choice == 'simple':
        return draw(expr_trees(max_depth=0))
    else:
        left = draw(expr_trees(max_depth=max_depth-1))
        right = draw(expr_trees(max_depth=max_depth-1))
        return left + right

@given(expr_trees(), expr_trees(), expr_trees())
def test_addition_associative(a, b, c):
    left_assoc = (a + b) + c
    right_assoc = a + (b + c)
    assert left_assoc == right_assoc, f"Associativity violated: ({a} + {b}) + {c} = {left_assoc} != {right_assoc} = {a} + ({b} + {c})"

if __name__ == "__main__":
    test_addition_associative()