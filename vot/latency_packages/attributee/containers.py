

from collections.abc import Mapping, Iterable

from attributee import Attribute, AttributeException

class ReadonlyMapping(Mapping):

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

class Tuple(Attribute):

    def __init__(self, *types, separator=",", **kwargs):
        super().__init__(**kwargs)
        for t in types:
            if not isinstance(t, Attribute):
                raise AttributeException("Illegal base class {}".format(t))

        self._types = types
        self._separator = separator

    def coerce(self, value, context=None):
        if isinstance(value, str):
            value = value.split(self._separator)
        if isinstance(value, dict):
            value = value.values()
        if not isinstance(value, Iterable):
            raise AttributeException("Unable to value convert to list")
        if context is None:
            context = dict()
        return [t.coerce(x, dict(key=i)) for i, (x, t) in enumerate(zip(value, self._types))]

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __getitem__(self, key):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __setitem__(self, key, value):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def dump(self, value):
        return [t.dump(x) for x, t in zip(value, self._types)]

    @property
    def types(self):
        return tuple(self._types)

class List(Attribute):

    def __init__(self, contains, separator=",", **kwargs):
        assert isinstance(contains, Attribute)
        self._separator = separator
        self._contains = contains
        super().__init__(**kwargs)

    def coerce(self, value, context=None):
        if isinstance(value, str):
            value = [v.strip() for v in value.split(self._separator)]
        if isinstance(value, dict):
            value = value.values()
        if not isinstance(value, Iterable):
            raise AttributeException("Unable to convert value to list")
        if context is None:
            context = dict()
        else:
            context = dict(**context)
        return [self._contains.coerce(x, context) for i, x in enumerate(value)]

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __getitem__(self, key):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __setitem__(self, key, value):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def dump(self, value):
        return [self._contains.dump(x) for x in value]

    @property
    def contains(self):
        return self._contains

class Map(Attribute):

    def __init__(self, contains, container=dict, **kwargs):
        assert isinstance(contains, Attribute)
        self._contains = contains
        self._container = container
        super().__init__(**kwargs)

    def coerce(self, value, context=None):
        if not isinstance(value, Mapping):
            raise AttributeException("Unable to value convert to dict")
        container = self._container()
        if context is None:
            context = dict()
        else:
            context = dict(**context)
        for name, data in value.items():
            context["key"] = name
            container[name] = self._contains.coerce(data, context)
        return ReadonlyMapping(container)

    def __iter__(self):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __getitem__(self, key):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def __setitem__(self, key, value):
        # This is only here to avoid pylint errors for the actual attribute field
        raise NotImplementedError

    def dump(self, value):
        return {k: self._contains.dump(v) for k, v in value.items()}

    @property
    def contains(self):
        return self._contains