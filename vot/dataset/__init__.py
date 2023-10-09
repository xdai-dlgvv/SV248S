import os
import json
import glob

from abc import abstractmethod, ABC

from PIL.Image import Image
import numpy as np

from vot import ToolkitException
from vot.utilities import read_properties
from vot.region import parse

import cv2

class DatasetException(ToolkitException):
    pass

class Channel(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def frame(self, index):
        pass

    @abstractmethod
    def filename(self, index):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

class Frame(object):

    def __init__(self, sequence, index):
        self._sequence = sequence
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def sequence(self) -> 'Sequence':
        return self._sequence

    def channels(self):
        return self._sequence.channels()

    def channel(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

    def filename(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.filename(self._index)

    def image(self, channel=None):
        channelobj = self._sequence.channel(channel)
        if channelobj is None:
            return None
        return channelobj.frame(self._index)

    def groundtruth(self):
        return self._sequence.groundtruth(self._index)

    def tags(self, index = None):
        return self._sequence.tags(self._index)

    def values(self, index=None):
        return self._sequence.values(self._index)

class SequenceIterator(object):

    def __init__(self, sequence):
        self._position = 0
        self._sequence = sequence

    def __iter__(self):
        return self

    def __next__(self):
        if self._position >= len(self._sequence):
            raise StopIteration()
        index = self._position
        self._position += 1
        return Frame(self._sequence, index)

class InMemoryChannel(Channel):

    def __init__(self):
        super().__init__()
        self._images = []
        self._width = 0
        self._height = 0
        self._depth = 0

    def append(self, image):
        if isinstance(image, Image):
            image = np.asarray(image)

        if len(image.shape) == 3:
            height, width, depth = image.shape
        elif len(image.shape) == 2:
            height, width = image.shape
            depth = 1
        else:
            raise DatasetException("Illegal image dimensions")

        if self._width > 0:
            if not (self._width == width and self._height == height):
                raise DatasetException("Size of images does not match")
            if not self._depth == depth:
                raise DatasetException("Channels of images does not match")
        else:
            self._width = width
            self._height = height
            self._depth = depth

        self._images.append(image)

    @property
    def length(self):
        return len(self._images)

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        return self._images[index]

    @property
    def size(self):
        return self._width, self._height

    def filename(self, index):
        raise DatasetException("Sequence is available in memory, image files not available")

class PatternFileListChannel(Channel):

    def __init__(self, path, start=1, step=1, end=None):
        super().__init__()
        base, pattern = os.path.split(path)
        self._base = base
        self._pattern = pattern
        self.__scan(pattern, start, step, end)

    @property
    def base(self):
        return self._base

    @property
    def pattern(self):
        return self._pattern

    def __scan(self, pattern, start, step, end):

        extension = os.path.splitext(pattern)[1]
        if not extension in {'.jpg', '.png', '.tif', '.tiff'}:
            raise DatasetException("Invalid extension in pattern {}".format(pattern))

        i = start
        self._files = []

        fullpattern = os.path.join(self.base, pattern)

        while True:
            image_file = os.path.join(fullpattern % i)

            if not os.path.isfile(image_file):
                break
            self._files.append(os.path.basename(image_file))
            i = i + step

            if end is not None and i > end:
                break

        if i <= start:
            raise DatasetException("Empty sequence, no frames found. %s" % fullpattern)

        im = cv2.imread(self.filename(0))
        self._width = im.shape[1]
        self._height = im.shape[0]
        self._depth = im.shape[2]

    @property
    def length(self):
        return len(self._files)

    def frame(self, index):
        if index < 0 or index >= self.length:
            return None

        bgr = cv2.imread(self.filename(index))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @property
    def size(self):
        return self._width, self._height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def filename(self, index):
        if index < 0 or index >= self.length:
            return None

        return os.path.join(self.base, self._files[index])

class FrameList(ABC):

    def __iter__(self):
        return SequenceIterator(self)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def frame(self, index) -> Frame:
        pass

class Sequence(FrameList):

    def __init__(self, name: str, dataset: "Dataset" = None):
        self._name = name
        self._dataset = dataset

    def __len__(self) -> int:
        return self.length

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._name

    @property
    def dataset(self):
        return self._dataset

    @abstractmethod
    def metadata(self, name, default=None):
        pass

    @abstractmethod
    def channel(self, channel=None):
        pass

    @abstractmethod
    def channels(self):
        pass

    @abstractmethod
    def groundtruth(self, index: int):
        pass

    @abstractmethod
    def tags(self, index=None):
        pass

    @abstractmethod
    def values(self, index=None):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @property
    @abstractmethod
    def length(self):
        pass

    def describe(self):
        data = dict(length=self.length, size=self.size)
        return data

class Dataset(ABC):

    def __init__(self, path):
        self._path = path

    def __len__(self):
        return self.length

    @property
    def path(self):
        return self._path

    @property
    @abstractmethod
    def length(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        return False

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def list(self):
        return []

    def keys(self):
        return self.list()

class BaseSequence(Sequence):

    def __init__(self, name, dataset=None):
        super().__init__(name, dataset)
        self._metadata = self._read_metadata()
        self._data = None

    @abstractmethod
    def _read_metadata(self):
        raise NotImplementedError

    @abstractmethod
    def _read(self):
        raise NotImplementedError

    def __preload(self):
        if self._data is None:
            self._data = self._read()

    def metadata(self, name, default=None):
        return self._metadata.get(name, default)

    def channels(self):
        self.__preload()
        return self._data[0]

    def channel(self, channel=None):
        self.__preload()
        if channel is None:
            channel = self.metadata("channel.default")
        return self._data[0].get(channel, None)

    def frame(self, index):
        return Frame(self, index)

    def groundtruth(self, index=None):
        self.__preload()
        if index is None:
            return self._data[1]
        return self._data[1][index]

    def tags(self, index=None):
        self.__preload()
        if index is None:
            return self._data[2].keys()
        return [t for t, sq in self._data[2].items() if sq[index]]

    def values(self, index=None):
        self.__preload()
        if index is None:
            return self._data[3].keys()
        return {v: sq[index] for v, sq in self._data[3].items()}

    @property
    def size(self):
        return self.channel().size

    @property
    def width(self):
        return self.channel().width

    @property
    def height(self):
        return self.channel().height

    @property
    def length(self):
        self.__preload()
        return len(self._data[1])

class InMemorySequence(BaseSequence):

    def __init__(self, name, channels):
        super().__init__(name, None)
        self._channels = {c: InMemoryChannel() for c in channels}
        self._tags = {}
        self._values = {}
        self._groundtruth = []

    def _read_metadata(self):
        return dict()

    def _read(self):
        return self._channels, self._groundtruth, self._tags, self._values

    def append(self, images: dict, region: "Region", tags: list = None, values: dict = None):

        if not set(images.keys()).issuperset(self._channels.keys()):
            raise DatasetException("Images not provided for all channels")

        for k, channel in self._channels.items():
            channel.append(images[k])

        if tags is None:
            tags = set()
        else:
            tags = set(tags)
        for tag in tags:
            if not tag in self._tags:
                self._tags[tag] = [False] * self.length
            self._tags[tag].append(True)
        for tag in set(self._tags.keys()).difference(tags):
                self._tags[tag].append(False)

        if values is None:
            values = dict()
        for name, value in values.items():
            if not name in self._values:
                self._values[name] = [0] * self.length
            self._values[tag].append(value)
        for name in set(self._values.keys()).difference(values.keys()):
                self._values[name].append(0)

        self._groundtruth.append(region)


from .vot import VOTDataset, VOTSequence
from .got10k import GOT10kSequence, GOT10kDataset
from .sv248s import SV248SSequence, SV248SDataset

def download_dataset(identifier: str, path: str):

    split = identifier.find(":")
    domain = "vot"

    if split > 0:
        domain = identifier[0:split].lower()
        identifier = identifier[split+1:]

    if domain == "vot":
        from .vot import download_dataset
        download_dataset(identifier, path)
    elif domain == "otb":
        from .otb import download_dataset
        download_dataset(path, identifier == "otb50")
    elif domain == 'ipiu':
        from .ipiu import download_dataset
        download_dataset(identifier, path)
    else:
        raise DatasetException("Unknown dataset domain: {}".format(domain))

def load_dataset(path: str):

    if not os.path.isdir(path):
        raise DatasetException("Dataset directory does not exist")

    if SV248SDataset.check(path):
        return SV248SDataset(path)
    else:
        raise DatasetException("Unsupported dataset type")

def load_sequence(path: str):
    if SV248SSequence.check(path):
        return SV248SSequence(path)
    else:
        raise DatasetException("Unsupported sequence type")