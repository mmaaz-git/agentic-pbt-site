def safe_type_fixed(arg, context=None):
    """Fixed version of safe_type that handles None context properly"""
    import sys

    py_type = type(arg)
    if py_type in (list, tuple, dict, str):
        return py_type.__name__
    elif py_type is complex:
        return 'double complex'
    elif py_type is float:
        return 'double'
    elif py_type is bool:
        return 'bint'
    elif 'numpy' in sys.modules and isinstance(arg, sys.modules['numpy'].ndarray):
        return 'numpy.ndarray[numpy.%s_t, ndim=%s]' % (arg.dtype.name, arg.ndim)
    else:
        for base_type in py_type.__mro__:
            if base_type.__module__ in ('__builtin__', 'builtins'):
                return 'object'
            # THIS IS THE FIX: Check if context is not None before calling find_module
            module = context.find_module(base_type.__module__, need_pxd=False) if context else None
            if module:
                entry = module.lookup(base_type.__name__)
                if entry.is_type:
                    return '%s.%s' % (base_type.__module__, base_type.__name__)
        return 'object'

# Test the fixed version
class CustomClass:
    pass

obj = CustomClass()

print("Testing fixed safe_type with custom class (context=None):")
result = safe_type_fixed(obj, context=None)
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Test with various types to ensure nothing broke
test_values = [
    42,
    3.14,
    True,
    [1, 2, 3],
    {'a': 1},
    "hello",
    3+4j,
    CustomClass(),
]

print("\nTesting fixed safe_type with all types:")
for val in test_values:
    result = safe_type_fixed(val)
    print(f"  {type(val).__name__:15s}: {result}")