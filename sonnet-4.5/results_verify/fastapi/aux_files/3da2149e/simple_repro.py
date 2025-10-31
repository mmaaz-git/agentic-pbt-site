import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr


class BuggyValidator:
    def __call__(self, inst, attr_obj, value):
        undefined_variable  # This should raise NameError


validator = attr.validators.or_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)

field_attr = attr.Attribute(
    name='test',
    default=attr.NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    eq=True,
    eq_key=None,
    order=False,
    order_key=None,
    hash=None,
    init=True,
    metadata={},
    type=None,
    converter=None,
    kw_only=False,
    inherited=False,
    on_setattr=None,
    alias=None
)

print("Testing or_ validator with buggy validator...")
try:
    validator(None, field_attr, 42)
    print("Result: No exception raised - NameError was silently caught!")
except NameError as e:
    print(f"Result: NameError was properly raised: {e}")
except Exception as e:
    print(f"Result: Different exception raised: {type(e).__name__}: {e}")