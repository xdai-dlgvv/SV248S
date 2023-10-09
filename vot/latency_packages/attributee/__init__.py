
import inspect
from typing import Type

from collections import OrderedDict
from collections.abc import Mapping

class AttributeException(Exception):
    pass

class AttributeParseException(AttributeException):
    def __init__(self, cause, key):
        self._keys = []
        if isinstance(cause, AttributeParseException):
            self._keys.extend(cause._keys)
            cause = cause.__cause__ or cause.__context__
        super().__init__(cause)
        self._keys.insert(0, key)
 
    def __str__(self):
        return "Attribute error: {}".format(".".join(self._keys))

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class Undefined():
    pass

def is_undefined(a):
    if a is None:
        return False
    return a == Undefined()

def is_instance_or_subclass(val, class_) -> bool:
    """Return True if ``val`` is either a subclass or instance of ``class_``."""
    try:
        return issubclass(val, class_)
    except TypeError:
        return isinstance(val, class_)

class Attribute(object):

    def __init__(self, default=Undefined(), description=""):
        self._default = default if is_undefined(default) else (None if default is None else self.coerce(default, {}))
        self._description = description

    def coerce(self, value, _):
        return value

    def dump(self, value):
        return value

    @property
    def default(self):
        return self._default

    @property
    def description(self):
        return self._description

    @property
    def required(self):
        return is_undefined(self._default)

class Any(Attribute):

    pass

class Nested(Attribute):

    def __init__(self, acls: Type["Attributee"], override: Mapping = None, **kwargs):
        if not issubclass(acls, Attributee):
            raise AttributeException("Illegal base class {}".format(acls))

        self._acls = acls
        self._override = dict(override.items() if not override is None else [])
        if "default" not in kwargs:
            self._required = False

            for _, afield in getattr(acls, "_declared_attributes", {}).items():
                if afield.required:
                    self._required = True
            if not self._required:
                kwargs["default"] = {}
        else:
            self._required = False

        super().__init__(**kwargs)

    def coerce(self, value, _):
        if value is None:
            return None
        assert isinstance(value, Mapping)
        kwargs = dict(value.items())
        kwargs.update(self._override)
        return self._acls(**kwargs)

    def dump(self, value: "Attributee"):
        if value is None:
            return None
        return value.dump()

    def attributes(self):
        return self._acls.attributes()

    @property
    def required(self):
        return super().required and self._required

    def __getattr__(self, name):
        # This is only here to avoid pylint errors for the actual attribute field
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        # This is only here to avoid pylint errors for the actual attribute field
        super().__setattr__(name, value)

class AttributeeMeta(type):

    @staticmethod
    def _get_fields(attrs: Mapping, pop=False):
        """Get fields from a class.
        :param attrs: Mapping of class attributes
        """
        fields = []
        for field_name, field_value in attrs.items():
            if is_instance_or_subclass(field_value, Attribute):
                fields.append((field_name, field_value))
        if pop:
            for field_name, _ in fields:
                del attrs[field_name]

        return fields

    # This function allows Schemas to inherit from non-Schema classes and ensures
    #   inheritance according to the MRO
    @staticmethod
    def _get_fields_by_mro(klass):
        """Collect fields from a class, following its method resolution order. The
        class itself is excluded from the search; only its parents are checked. Get
        fields from ``_declared_attributes`` if available, else use ``__dict__``.

        :param type klass: Class whose fields to retrieve
        """
        mro = inspect.getmro(klass)
        # Loop over mro in reverse to maintain correct order of fields
        return sum(
            (
                AttributeeMeta._get_fields(
                    getattr(base, "_declared_attributes", base.__dict__)
                )
                for base in mro[:0:-1]
            ),
            [],
        )

    @classmethod
    def __prepare__(self, name, bases):
        return OrderedDict()

    def __new__(mcs, name, bases, attrs):

        cls_attributes = AttributeeMeta._get_fields(attrs, pop=True)
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_attributes = AttributeeMeta._get_fields_by_mro(klass)

        # Assign attributes on class
        klass._declared_attributes = OrderedDict(inherited_attributes + cls_attributes)

        return klass

class Include(Nested):

    def filter(self, **kwargs):
        attributes = getattr(self._acls, "_declared_attributes", {})
        filtered = dict()
        for aname, afield in attributes.items():
            if isinstance(afield, Include):
                filtered.update(afield.filter(**kwargs))
            elif aname in kwargs:
                filtered[aname] = kwargs[aname]
        return filtered

class Attributee(metaclass=AttributeeMeta):

    def __init__(self, *args, **kwargs):
        super().__init__()
        attributes = getattr(self.__class__, "_declared_attributes", {})

        unconsumed = set(kwargs.keys())
        unspecified = set(attributes.keys())

        for avalue, aname in zip(args, filter(lambda x: not isinstance(attributes[x], Include) and x not in kwargs, attributes.keys())):
            if aname in kwargs:
                raise AttributeException("Argument defined as positional and keyword: {}".format(aname))
            kwargs[aname] = avalue

        for aname, afield in attributes.items():
            try:
                if isinstance(afield, Include):
                    iargs = afield.filter(**kwargs)
                    super().__setattr__(aname, afield.coerce(iargs, {"parent": self}))
                    unconsumed.difference_update(iargs.keys())
                    unspecified.difference_update(iargs.keys())
                else:
                    if not aname in kwargs:
                        if not afield.required:
                            avalue = afield.default
                            super().__setattr__(aname, avalue)
                        else:
                            continue
                    else:
                        avalue = kwargs[aname]
                        try:
                            value = afield.coerce(avalue, {"parent": self})
                            super().__setattr__(aname, value)
                        except AttributeException as ae:
                            raise AttributeParseException(ae, aname) from ae
                        except AttributeError as ae:
                            raise AttributeParseException(ae, aname) from ae
            except AttributeError:
                raise AttributeException("Illegal attribute name {}, already taken".format(aname))
            unconsumed.difference_update([aname])
            unspecified.difference_update([aname])

        if unspecified:
            raise AttributeException("Missing arguments: {}".format(", ".join(unspecified)))

        if unconsumed:
            raise AttributeException("Unsupported arguments: {}".format(", ".join(unconsumed)))

    def __setattr__(self, key, value):
        attributes = getattr(self.__class__, "_declared_attributes", {})
        if key in attributes:
            raise AttributeException("Attribute {} is readonly".format(key))
        super().__setattr__(key, value)

    @classmethod
    def attributes(cls):
        from .containers import ReadonlyMapping
        attributes = getattr(cls, "_declared_attributes", {})
        return ReadonlyMapping(attributes)

    def dump(self, ignore=None):
        attributes = getattr(self.__class__, "_declared_attributes", {})
        if attributes is None:
            return OrderedDict()
    
        serialized = OrderedDict()
        for aname, afield in attributes.items():
            if ignore is not None and aname in ignore:
                continue
            if isinstance(afield, Include):
                serialized.update(afield.dump(getattr(self, aname, {})))
            else:
                serialized[aname] = afield.dump(getattr(self, aname, afield.default))
                
        return serialized

    @classmethod
    def list_attributes(cls):
        for name, attr in cls.attributes().items():
            yield (name, attr)

from attributee.primitives import Integer, Float, String, Boolean, Enumeration, Primitive, Number
from attributee.object import Object, Callable, Date, Datetime
from attributee.containers import List, Map, Tuple