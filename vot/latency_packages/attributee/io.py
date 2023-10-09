
import os
import sys
import typing
import collections
import argparse
from functools import partial

from attributee import Attributee, AttributeException, Include, is_undefined, Boolean, Nested

def _dump_serialized(obj: Attributee, handle: typing.Union[typing.IO[str], str], dumper: typing.Callable):
    data = obj.dump()

    if isinstance(handle, str):
        with open(handle, "w") as stream:
            dumper(data, stream)
    else:
        dumper(data, handle)

def _load_serialized(handle: typing.Union[typing.IO[str], str], factory: typing.Callable, loader: typing.Callable):
    if isinstance(handle, str):
        with open(handle, "r") as stream:
            data = loader(stream)
    else:
        data = loader(handle)

    return factory(**data)

try:

    import yaml

    def _yaml_load(stream):
        class OrderedLoader(yaml.Loader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return collections.OrderedDict(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    def _yaml_dump(data, stream=None, **kwds):
        class OrderedDumper(yaml.Dumper):
            pass
        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items())
        OrderedDumper.add_representer(collections.OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwds)

    dump_yaml = partial(_dump_serialized, dumper=_yaml_dump)
    load_yaml = partial(_load_serialized, loader=_yaml_load)

except ImportError:

    def _no_support():
        raise ImportError("PyYAML not installed")

    dump_yaml = lambda a, b: _no_support()
    load_yaml = lambda a, b: _no_support()
    pass


import json

dump_json = partial(_dump_serialized, dumper=partial(json.dump))
load_json = partial(_load_serialized, loader=partial(json.load, object_pairs_hook=collections.OrderedDict))




class _StorePrefix(argparse.Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)
 
    def _route_dest(self, namespace, values):
        path = self.dest.split(".")
        if len(path) > 1:
            container = getattr(namespace, path[0], {})
            for key in path[1:-1]:
                container = container.setdefault(key, {})
            container[path[-1]] = values
            values = container
            dest = path[0]
        else:
            dest = self.dest
        setattr(namespace, dest, values)

    def __call__(self, parser, namespace, values, option_string=None):
        self._route_dest(namespace, values)

class _StoreTrueAction(_StorePrefix):
    
    def __init__(self, option_strings, dest, default=None, required=False, help=None, metavar=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=True,
            default=default,
            required=required,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        self._route_dest(namespace, self.const)

class _StoreFalseAction(_StorePrefix):

    def __init__(self, option_strings, dest, default=None, required=False, help=None, metavar=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=False,
            default=default,
            required=required,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        self._route_dest(namespace, self.const)

class Entrypoint(object):
    """ A mixin that provides initialization of Attributee object using command line arguments.

    """

    @classmethod
    def parse(cls, boolean_flags=None):
        """[summary]

        Args:
            boolean_flags (bool, optional): [description]. Defaults to True.

        Raises:
            AttributeException: [description]

        Returns:
            [type]: [description]
        """
        if not issubclass(cls, Attributee):
            raise AttributeException("Not a valid base class")

        if boolean_flags is None:
            boolean_flags = not os.environ.get("ATTIRBUTEE_ARGPARSE_BOOLEAN", "false").lower() in ("true", "1")

        args = dict()

        parser = argparse.ArgumentParser(conflict_handler='resolve')

        def add_arguments(parser: argparse.ArgumentParser, attributes, prefixes = None):
            for name, attr in attributes.items():
                data = {}
                
                prefix = "" if prefixes is None else ".".join(prefixes) + "."
                data["action"] = _StorePrefix
                data["dest"] = prefix + name
                data["default"] = argparse.SUPPRESS

                if isinstance(attr, Nested):
                    if isinstance(attr, Include):
                        add_arguments(parser, attr.attributes(), prefixes)
                    else:
                        prefixes = [name] if prefixes is None else prefixes + [name]
                        add_arguments(parser, attr.attributes(), prefixes)
                    continue
                elif isinstance(attr, Boolean) and boolean_flags:
                    if not is_undefined(attr.default) and attr.default is True:
                        data["action"] = _StoreFalseAction
                        name = "not_" + name
                    else:
                        data["action"] = _StoreTrueAction
                    data["required"] = False
                elif not is_undefined(attr.default):
                    data["required"] = False
                else:
                    data["required"] = True
                if attr.description is not None:
                    data["help"] = attr.description

                parser.add_argument("--" + prefix + name, **data)

        add_arguments(parser, cls.attributes())

        args = parser.parse_args()

        return cls(**vars(args))

class Serializable(object):
    """ A mixin that provides handy IO methods for Attributtee derived classes.

    """

    @classmethod
    def read(cls, source: str):
        if not issubclass(cls, Attributee):
            raise AttributeException("Not a valid base class")
        ext = os.path.splitext(source)[1].lower()
        if ext in [".yml", ".yaml"]:
            return load_yaml(source, cls)
        if ext in [".json"]:
            return load_json(source, cls)
        else:
            raise AttributeException("Unknown file format")


    def write(self, destination: str):
        if not isinstance(self, Attributee):
            raise AttributeException("Not a valid base class")
        ext = os.path.splitext(destination)[1].lower()
        if ext in [".yml", ".yaml"]:
            return dump_yaml(self, destination)
        if ext in [".json"]:
            return dump_json(self, destination)
        else:
            raise AttributeException("Unknown file format")


