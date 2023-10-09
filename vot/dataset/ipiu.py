
import os
import glob
import logging
from collections import OrderedDict

import six

import cv2

from vot.dataset import Dataset, DatasetException, Sequence, BaseSequence, PatternFileListChannel
from vot.region import parse, write_file
from vot.utilities import Progress, localize_path, read_properties, write_properties

logger = logging.getLogger("vot")

def load_channel(source):

    extension = os.path.splitext(source)[1]

    if extension == '':
        source = os.path.join(source, '%08d.jpg')
    return PatternFileListChannel(source)

class VOTSequence(BaseSequence):

    def __init__(self, base, name=None, dataset=None):
        self._base = base
        if name is None:
            name = os.path.basename(base)
        super().__init__(name, dataset)

    @staticmethod
    def check(path):
        return os.path.isfile(os.path.join(path, 'sequence'))

    def _read_metadata(self):
        metadata = dict(fps=30, format="default")
        metadata["channel.default"] = "color"

        metadata_file = os.path.join(self._base, 'sequence')
        metadata.update(read_properties(metadata_file))

        return metadata

    def _read(self):

        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        for c in ["color", "depth", "ir"]:
            channel_path = self.metadata("channels.%s" % c, None)
            if not channel_path is None:
                channels[c] = load_channel(os.path.join(self._base, localize_path(channel_path)))

        # Load default channel if no explicit channel data available
        if len(channels) == 0:
            channels["color"] = load_channel(os.path.join(self._base, "color", "%08d.jpg"))
        else:
            self._metadata["channel.default"] = next(iter(channels.keys()))

        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        groundtruth_file = os.path.join(self._base, self.metadata("groundtruth", "groundtruth.txt"))

        with open(groundtruth_file, 'r') as filehandle:
            for region in filehandle.readlines():
                groundtruth.append(parse(region))

        self._metadata["length"] = len(groundtruth)

        tagfiles = glob.glob(os.path.join(self._base, '*.tag')) + glob.glob(os.path.join(self._base, '*.label'))

        for tagfile in tagfiles:
            with open(tagfile, 'r') as filehandle:
                tagname = os.path.splitext(os.path.basename(tagfile))[0]
                tag = [line.strip() == "1" for line in filehandle.readlines()]
                while not len(tag) >= len(groundtruth):
                    tag.append(False)
                tags[tagname] = tag

        valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        for valuefile in valuefiles:
            with open(valuefile, 'r') as filehandle:
                valuename = os.path.splitext(os.path.basename(valuefile))[0]
                value = [float(line.strip()) for line in filehandle.readlines()]
                while not len(value) >= len(groundtruth):
                    value.append(0.0)
                values[valuename] = value

        for name, channel in channels.items():
            if not channel.length == len(groundtruth):
                raise DatasetException("Length mismatch for channel %s (%d != %d)" % (name, channel.length, len(groundtruth)))

        for name, tag in tags.items():
            if not len(tag) == len(groundtruth):
                tag_tmp = len(groundtruth) * [False]
                tag_tmp[:len(tag)] = tag
                tag = tag_tmp

        for name, value in values.items():
            if not len(value) == len(groundtruth):
                raise DatasetException("Length mismatch for value %s" % name)

        return channels, groundtruth, tags, values

class VOTDataset(Dataset):

    def __init__(self, path):
        super().__init__(path)

        if not os.path.isfile(os.path.join(path, "list.txt")):
            raise DatasetException("Dataset not available locally")

        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(names)) as progress:

            for name in names:
                self._sequences[name.strip()] = VOTSequence(os.path.join(path, name.strip()), dataset=self)
                progress.relative(1)

    @staticmethod
    def check(path: str):
        if not os.path.isfile(os.path.join(path, 'list.txt')):
            return False

        with open(os.path.join(path, 'list.txt'), 'r') as handle:
            sequence = handle.readline().strip()
            return VOTSequence.check(os.path.join(path, sequence))

    @property
    def path(self):
        return self._path

    @property
    def length(self):
        return len(self._sequences)

    def __getitem__(self, key):
        return self._sequences[key]

    def __contains__(self, key):
        return key in self._sequences

    def __iter__(self):
        return self._sequences.values().__iter__()

    def list(self):
        return list(self._sequences.keys())

    @classmethod
    def download(self, url, path="."):
        print("Please verifiy the sequence.")
        return True

def write_sequence(directory: str, sequence: Sequence):

    channels = sequence.channels()

    metadata = dict()
    metadata["channel.default"] = sequence.metadata("channel.default", "color")
    metadata["fps"] = sequence.metadata("fps", "30")

    for channel in channels:
        cdir = os.path.join(directory, channel)
        os.makedirs(cdir, exist_ok=True)

        metadata["channels.%s" % channel] = os.path.join(channel, "%08d.jpg")

        for i in range(sequence.length):
            frame = sequence.frame(i).channel(channel)
            cv2.imwrite(os.path.join(cdir, "%08d.jpg" % (i + 1)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for tag in sequence.tags():
        data = "\n".join(["1" if tag in sequence.tags(i) else "0" for i in range(sequence.length)])
        with open(os.path.join(directory, "%s.tag" % tag), "w") as fp:
            fp.write(data)

    for value in sequence.values():
        data = "\n".join([ str(sequence.values(i).get(value, "")) for i in range(sequence.length)])
        with open(os.path.join(directory, "%s.value" % value), "w") as fp:
            fp.write(data)

    write_file(os.path.join(directory, "groundtruth.txt"), [f.groundtruth() for f in sequence])
    write_properties(os.path.join(directory, "sequence"), metadata)


VOT_DATASETS = {
    "label102" : ""
}

def download_dataset(name, path="."):
    if not name in VOT_DATASETS:
        raise ValueError("Unknown dataset")
    VOTDataset.download(VOT_DATASETS[name], path)