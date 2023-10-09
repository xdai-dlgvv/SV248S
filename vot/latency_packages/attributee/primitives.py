
import inspect
import typing
from collections.abc import Mapping
from enum import Enum

from attributee import Attribute, AttributeException

def _parse_number(value):
    if isinstance(value, int) or isinstance(value, float):
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)

def to_string(n):
    if n is None:
        return ""
    else:
        return str(n)

def to_number(val, max_n = None, min_n = None, conversion=_parse_number):
    try:
        n = conversion(val)

        if not max_n is None:
            if n > max_n:
                raise AttributeException("Parameter higher than maximum allowed value ({}>{})".format(n, max_n))
        if not min_n is None:
            if n < min_n:
                raise AttributeException("Parameter lower than minimum allowed value ({}<{})".format(n, min_n))

        return n
    except ValueError as ve:
        raise AttributeException("Number conversion error") from ve

def to_logical(val):
    try:
        if isinstance(val, str):
            return val.lower() in ['true', '1', 't', 'y', 'yes']
        else:
            return bool(val)

    except ValueError as ve:
        raise AttributeException("Logical value conversion error") from ve

class Primitive(Attribute):

    def coerce(self, value, _):
        assert value is None or isinstance(value, (str, int, bool, float))
        return value

class Number(Attribute):

    def __init__(self, conversion=_parse_number, val_min=None, val_max=None, **kwargs):
        self._conversion = conversion
        self._val_min = conversion(val_min) if val_min is not None else None
        self._val_max = conversion(val_max) if val_max is not None else None
        super().__init__(**kwargs)

    def coerce(self, value, _=None):
        return to_number(value, max_n=self._val_max, min_n=self._val_min, conversion=self._conversion)

    @property
    def min(self):
        return self._val_min

    @property
    def max(self):
        return self._val_max

class Integer(Number):

    def __init__(self, **kwargs):
        super().__init__(conversion=int, **kwargs)

class Float(Number):

    def __init__(self, **kwargs):
        super().__init__(conversion=float, **kwargs)

class Boolean(Attribute):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def coerce(self, value, _):
        return to_logical(value)

class String(Attribute):

    def __init__(self, transformer=None, **kwargs):
        self._transformer = transformer
        super().__init__(**kwargs)

    def coerce(self, value, ctx):
        if value is None:
            return None
        if self._transformer is None:
            return to_string(value)
        else:
            return self._transformer(to_string(value), ctx)

    @property
    def transformer(self):
        return self._transformer

class Enumeration(Attribute):

    def __init__(self, options,  **kwargs):
        if inspect.isclass(options) and issubclass(options, Enum):
            self._mapping = options
        elif isinstance(options, Mapping):
            self._mapping = options
            self._inverse = {v: k for k, v in options.items()}
        else:
            raise AttributeException("Not an enum class or dictionary")
        super().__init__(**kwargs)

    def coerce(self, value, ctx):
        if isinstance(value, (str, int, float)):
            if inspect.isclass(self._mapping) and issubclass(self._mapping, Enum):
                return self._mapping(value)
            else:
                return self._mapping[value]
        elif inspect.isclass(self._mapping) and isinstance(value, self._mapping):
            return value
        else:
            raise AttributeException("Cannot parse enumeration: {}".format(value))

    def dump(self, value):
        if inspect.isclass(self._mapping) and isinstance(value, self._mapping):
            return value.value
        else:
            return self._inverse[value]

    @property
    def options(self):
        from .containers import ReadonlyMapping
        if inspect.isclass(self._mapping) and issubclass(self._mapping, Enum):
            return ReadonlyMapping([(e.value, e.name) for e in self._mapping])
        elif isinstance(self._mapping, typing.Mapping):
            return ReadonlyMapping(self._mapping) 

__all__ = ["String", "Boolean", "Integer", "Float", "Enumeration", "Number"]