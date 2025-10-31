"""
Demonstrates the bug in Django's TemplateCommand.validate_name method.

The method uses find_spec() to check if a name conflicts with an existing module,
but find_spec('__main__') raises ValueError instead of returning None or a spec.

While '__main__' doesn't pass the isidentifier() check and thus wouldn't normally
reach find_spec, the bug exists in the error handling logic and could be triggered
in edge cases or subclasses.
"""

from django.core.management.templates import TemplateCommand
from importlib.util import find_spec

# First demonstrate that find_spec('__main__') raises ValueError
print("1. Testing find_spec('__main__') directly:")
try:
    spec = find_spec('__main__')
    print(f"   Result: {spec}")
except ValueError as e:
    print(f"   ValueError raised: {e}")

print()

# Now demonstrate the bug - we'll subclass to bypass isidentifier check
class BuggyTemplateCommand(TemplateCommand):
    def validate_name(self, name, name_or_dir="name"):
        """Override to bypass isidentifier check and expose find_spec bug"""
        if name is None:
            from django.core.management.base import CommandError
            raise CommandError(
                "you must provide {an} {app} name".format(
                    an=self.a_or_an,
                    app=self.app_or_project,
                )
            )
        # Skip isidentifier check - go straight to find_spec
        # This is where the bug is - find_spec can raise ValueError
        from importlib.util import find_spec
        if find_spec(name) is not None:
            from django.core.management.base import CommandError
            raise CommandError(
                "'{name}' conflicts with the name of an existing Python "
                "module and cannot be used as {an} {app} {type}. Please try "
                "another {type}.".format(
                    name=name,
                    an=self.a_or_an,
                    app=self.app_or_project,
                    type=name_or_dir,
                )
            )

print("2. Testing BuggyTemplateCommand.validate_name('__main__'):")
cmd = BuggyTemplateCommand()
cmd.app_or_project = 'app'
cmd.a_or_an = 'an'

try:
    cmd.validate_name('__main__', 'name')
    print("   No error occurred")
except ValueError as e:
    print(f"   BUG: ValueError raised instead of CommandError: {e}")
except Exception as e:
    print(f"   {e.__class__.__name__}: {e}")