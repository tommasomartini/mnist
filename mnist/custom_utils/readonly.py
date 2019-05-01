class ReadOnlyError(Exception):
    """An exception to raise when trying to edit a constant class."""
    pass


def _setattr_override(obj, name, value):
    """Function used to override a class __setattr__ method. If called,
    raises an exception.

    Args:
        obj: Object whose attribute is to be set.
        name: Name of the attribute.
        value: Value of the attribute.

    Raises:
        ReadOnlyError: Whenever this function is called.
    """
    raise ReadOnlyError('Class {} is ReadOnly: attribute {} cannot be '
                        'edited or added'.format(obj.__name__, name))


def _delattr_override(obj, name):
    """Function used to override a class __delattr__ method. If called,
    raises an exception.

    Args:
        obj: Object whose attribute is to be set.
        name: Name of the attribute.

    Raises:
        ReadOnlyError: Whenever this function is called.
    """
    raise ReadOnlyError('Class {} is ReadOnly: attribute {} cannot be '
                        'deleted'.format(obj.__name__, name))


class _ReadOnlyMeta(type):
    """All the classes extending from `ReadOnly` will inherit the read-only
    property thanks to this meta-class.
    """
    __setattr__ = _setattr_override
    __delattr__ = _delattr_override


class ReadOnly(metaclass=_ReadOnlyMeta):
    """Attributes of subclasses of `ReadOnly` cannot be added, modified or
    deleted. Moreover, no new attributes can be added.

    Examples:
        >>> class MyClass(ReadOnly):
        >>>     A = 10

        >>> MyClass.A = 20  # FAIL: edit existing attribute
        >>> MyClass.B = 20  # FAIL: create new attribute
        >>> del MyClass.A   # FAIL: delete existing attribute
    """
    pass
