import contextvars
import importlib
from contextlib import contextmanager
from typing import Any, Type


def get_class_from_str(class_str: str, package: str | None = None) -> Type[Any]:
    """
    Converts a string to the corresponding class object, supporting relative imports.
    For relative module paths (starting with '.'), a package must be provided.

    Args:
        class_str: String representation of the class, either absolute or relative.
        package: Package context, only required for relative imports.

    Returns: Class object corresponding to the provided string.
    """
    if not isinstance(class_str, str) and isinstance(class_str, type):
        return class_str

    module_path, _, class_name = class_str.rpartition(".")
    if not module_path and class_str.startswith("."):
        module_path = "."
    if module_path.startswith("."):
        if not package:
            raise ValueError("Relative module path provided without a package context.")
        module = importlib.import_module(module_path, package=package)
    else:
        module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_str_from_class(cls: Type[Any], package: str | None = None) -> str:
    """
    Converts a class object to its string representation.
    If a package is provided and the class's module is a submodule of the package,
    the returned string will use a relative import.
    Otherwise, an absolute import string is returned.

    Args:
        cls: Class object to convert.
        package: Package context, only required for relative imports.

    Returns: String representation of the class.
    """
    if isinstance(cls, str):
        return cls

    module_path = cls.__module__
    class_name = cls.__name__

    if package:
        # When class is defined directly in the package's __init__.py
        if module_path == package:
            return f".{class_name}"
        # When class is in a submodule of the package
        elif module_path.startswith(package + "."):
            # Get the relative part (including the dot)
            relative = module_path[len(package) :]
            if not relative.startswith("."):
                relative = "." + relative
            return f"{relative}.{class_name}"
    return f"{module_path}.{class_name}"


use_init_empty_weights = contextvars.ContextVar("init_empty_weights", default=False)


@contextmanager
def init_empty_weights(value: bool):
    """
    Context manager to indicate that a (parametrized) model should be initialized with empty weights or not.
    If active, `use_init_empty_weights` will be set to `True` otherwise to `False`.
    To check if the context is active, import and check `use_init_empty_weights.get()`.

    Args:
        value: Indicates whether the model should be initialized with empty weights or not.
    """
    token = use_init_empty_weights.set(value)
    try:
        yield
    finally:
        use_init_empty_weights.reset(token)
