"""Project package.

This package is named ``code``, which can shadow Python's stdlib ``code`` module.
PyCharm's debugger imports ``InteractiveConsole`` from ``code`` during startup.
Expose the stdlib symbols here so debugger imports still work.
"""

import importlib.util
import os
from types import ModuleType


def _load_stdlib_code_module() -> ModuleType:
    # Resolve the stdlib folder via the location of the stdlib os module.
    stdlib_dir = os.path.dirname(os.__file__)
    stdlib_code_path = os.path.join(stdlib_dir, "code.py")

    spec = importlib.util.spec_from_file_location("_stdlib_code", stdlib_code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stdlib code module from {stdlib_code_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_code = _load_stdlib_code_module()

# Re-export stdlib names expected by debugger tooling.
InteractiveConsole = _stdlib_code.InteractiveConsole
InteractiveInterpreter = _stdlib_code.InteractiveInterpreter
compile_command = _stdlib_code.compile_command

__all__ = ["InteractiveConsole", "InteractiveInterpreter", "compile_command"]

