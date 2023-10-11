
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

# def load_channel(source):

#     extension = os.path.splitext(source)[1]

#     # if extension == '':
#     #     source = os.path.join(source, '%08d.jpg')
#     return PatternFileListChannel(source)

class SV248SSequence(BaseSequence):

    def __init__(self, base, name=None, dataset=None):
        self.__scene, self.__name = os.path.basename(base).split('_')
        self._base = os.path.join(os.path.dirname(base), self.__scene)
        self._seq_dir = os.path.join(self._base, 'sequences', self.__name)
        self._anno_dir = os.path.join(self._base, 'annotations', self.__name)
        if name is None:
            name = os.path.basename(base)
        
        super().__init__(name, dataset)

    @staticmethod
    def check(path):
        # check the dataset's directories, must contain "sequences" and "annotations"
        return os.path.isdir(os.path.join(path, 'sequences')) and os.path.isdir(os.path.join(path, 'annotations')) 

    def _read_metadata(self):
        metadata = dict(fps=25, format="default")
        metadata["channel.default"] = "color"

        return metadata

    def _read(self):
        
        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        channels["color"] = PatternFileListChannel(os.path.join(self._seq_dir, "%06d.tiff"))

        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        groundtruth_file = self._anno_dir + '.poly'

        with open(groundtruth_file, 'r') as filehandle:
            for region in filehandle.readlines():
                groundtruth.append(parse(region))

        self._metadata["length"] = len(groundtruth)

        tagfile = self._anno_dir + '.state'

        tag_inv = [False] * len(groundtruth)
        tag_occ = [False] * len(groundtruth)

        with open(tagfile, 'r') as filehandle:
            lines = filehandle.readlines()
        for i, line in enumerate(lines):
            if line.strip() == '1':
                tag_inv[i] = True
            elif line.strip() == '2':
                tag_occ[i] = True
        tags['inv'] = tag_inv
        tags['occ'] = tag_occ

        # valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        # for valuefile in valuefiles:
        #     with open(valuefile, 'r') as filehandle:
        #         valuename = os.path.splitext(os.path.basename(valuefile))[0]
        #         value = [float(line.strip()) for line in filehandle.readlines()]
        #         while not len(value) >= len(groundtruth):
        #             value.append(0.0)
        #         values[valuename] = value

        for name, channel in channels.items():
            if not channel.length == len(groundtruth):
                raise DatasetException("Length mismatch for channel %s (%d != %d)" % (name, channel.length, len(groundtruth)))

        # for name, tag in tags.items():
        #     if not len(tag) == len(groundtruth):
        #         tag_tmp = len(groundtruth) * [False]
        #         tag_tmp[:len(tag)] = tag
        #         tag = tag_tmp

        # for name, value in values.items():
        #     if not len(value) == len(groundtruth):
        #         raise DatasetException("Length mismatch for value %s" % name)

        return channels, groundtruth, tags, values

class SV248SDataset(Dataset):

    @staticmethod
    def create_list_file_for_all_targets(path):
        list_all = []
        for scene in ['01', '02', '03', '04', '05', '06']:
            scene_dir = os.path.join(path, scene)
            seqs_dir = os.path.join(scene_dir, 'sequences')
            seqs = os.listdir(seqs_dir)
            for seq in seqs:
                seq_dir = os.path.join(seqs_dir, seq)
                if os.path.exists(seq_dir) and seq.isnumeric():
                    list_all.append("%s_%s\n" % (scene, seq))
        if len(list_all) != 248:
            logger.warn('The dataset has incomplete sequences, please check! The numbers should be 248\
                         for all sequences but we got %d' % len(list_all))
        with open(os.path.join(path, 'list.txt'), 'w') as f:
            f.writelines(list_all)
        logger.info('The full verison dataset list is saved: %s' % (os.path.join(path, 'list.txt')))
            
    
    def __init__(self, path):
        super().__init__(path)

        if not os.path.isfile(os.path.join(path, "list.txt")):
            # create new list.txt file to contain all the videos
            self.create_list_file_for_all_targets(path)

        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(names)) as progress:

            for name in names:
                self._sequences[name.strip()] = SV248SSequence(os.path.join(path, name.strip()), dataset=self)
                progress.relative(1)

    @staticmethod
    def check(path: str):
        # in SV248S dataset, there are 6 scene, numbered from 01~06
        for scene in ['01', '02', '03', '04', '05', '06']:
            if not os.path.isdir(os.path.join(path, scene)):
                raise DatasetException('Incomplete dataset: Lack of scene %s.' % scene)
            if not SV248SSequence.check(os.path.join(path, scene)):
                raise DatasetException('Incomplete sequence directories in %s' % os.path.join(path, scene))
        logger.info('SV248S Dataset Checked! [PASS]')
        return True

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
        raise DatasetException('Unsupported Operation in this dataset! Please download manually!')

# def write_sequence(directory: str, sequence: Sequence):

#     channels = sequence.channels()

#     metadata = dict()
#     metadata["channel.default"] = sequence.metadata("channel.default", "color")
#     metadata["fps"] = sequence.metadata("fps", "25")

#     for channel in channels:
#         cdir = os.path.join(directory, channel)
#         os.makedirs(cdir, exist_ok=True)

#         metadata["channels.%s" % channel] = os.path.join(channel, "%08d.jpg")

#         for i in range(sequence.length):
#             frame = sequence.frame(i).channel(channel)
#             cv2.imwrite(os.path.join(cdir, "%08d.jpg" % (i + 1)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     for tag in sequence.tags():
#         data = "\n".join(["1" if tag in sequence.tags(i) else "0" for i in range(sequence.length)])
#         with open(os.path.join(directory, "%s.tag" % tag), "w") as fp:
#             fp.write(data)

#     for value in sequence.values():
#         data = "\n".join([ str(sequence.values(i).get(value, "")) for i in range(sequence.length)])
#         with open(os.path.join(directory, "%s.value" % value), "w") as fp:
#             fp.write(data)

#     write_file(os.path.join(directory, "groundtruth.txt"), [f.groundtruth() for f in sequence])
#     write_properties(os.path.join(directory, "sequence"), metadata)


VOT_DATASETS = {
    "SV248S-results": ""
}

def download_dataset(name, path="."):
    raise DatasetException('Unsupported Operation in this dataset!')
    # if not name in VOT_DATASETS:
    #     raise ValueError("Unknown dataset")
    # VOTDataset.download(VOT_DATASETS[name], path)