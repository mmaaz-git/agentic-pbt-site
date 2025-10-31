import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# Let's import the internal parser
from numpy.f2py.symbolic import _FromStringWorker, replace_parenthesis, unreplace_parenthesis

class DebugFromStringWorker(_FromStringWorker):
    def process(self, s, context='expr'):
        """Debug version of process to trace what's happening"""
        if isinstance(s, (list, tuple)):
            return type(s)(self.process(s_, context) for s_ in s)

        assert isinstance(s, str), (type(s), s)

        # Replace subexpressions in parenthesis
        r, raw_symbols_map = replace_parenthesis(s)
        r = r.strip()

        def restore(r):
            if isinstance(r, (list, tuple)):
                return type(r)(map(restore, r))
            return unreplace_parenthesis(r, raw_symbols_map)

        print(f"\n--- Processing: '{s}' (context={context}) ---")
        print(f"After replace_parenthesis: r='{r}'")
        print(f"raw_symbols_map: {raw_symbols_map}")

        # Check if it goes to multiplication
        import re
        operands_mult = re.split(r'(?<=[@\w\d_])\s*([*]|/)',
                                (r if self.language is Language.C
                                 else r.replace('**', '@__f2py_DOUBLE_STAR@')))
        print(f"Multiplication split: {operands_mult}")

        if len(operands_mult) > 1:
            print("-> Goes to multiplication branch")
        else:
            # Check for exponentiation
            if self.language is not Language.C and '**' in r:
                print("-> Goes to exponentiation branch")
                operands = list(reversed(restore(r.split('**'))))
                print(f"   Exponentiation operands: {operands}")
            else:
                print("-> Does not go to exponentiation (no '**' found in r)")

        # Call the original process
        return super().process(s, context)

# Test with our debug worker
worker = DebugFromStringWorker(language=Language.Fortran)
result = worker.parse('x**2')
print(f"\nFinal result: {result}")
print(f"Final tostring: {result.tostring()}")