from typing import List, Tuple, Any

import numpy as np

from attributee import Boolean, Integer, Include

from vot.analysis import (Measure,
                          MissingResultsException,
                          SequenceAggregator, Sorting,
                          is_special, SeparableAnalysis,
                          analysis_registry)
from vot.dataset import Sequence
from vot.experiment import Experiment
from vot.experiment.multirun import (MultiRunExperiment, SupervisedExperiment)
from vot.region import Region, Special, calculate_overlaps, calculate_location_errs, calculate_ps, calculate_rs
from vot.tracker import Tracker
from vot.utilities.data import Grid
from vot.region.shapes import Rectangle

def compute_accuracy(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:


    gt_rect = []
    for gt_poly in sequence.groundtruth():
        if len(gt_rect) == 0:
            gt_rect.append(Special(1))
            continue
        points = np.array(gt_poly.points())
        x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
        gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))

    overlaps = np.array(calculate_overlaps(trajectory, gt_rect, (sequence.size) if bounded else None))

    # overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False
    
    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_OSLE(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
    else:
        gt_rect = sequence.groundtruth()

    overlaps = np.array(calculate_overlaps(trajectory, gt_rect, (sequence.size) if bounded else None))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0

def compute_Eprecision(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
    else:
        p_standard = 1
    #print(p_standard)
    overlaps = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    overlaps = 1 - np.power(overlaps/p_standard - 1, 2)
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_Erecall(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_rs(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
    else:
        p_standard = 1

    overlaps = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = 1 - np.power(overlaps/p_standard - 1, 2)

    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_recall(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
    else:
        gt_rect = sequence.groundtruth()

    overlaps = np.array(calculate_rs(trajectory, gt_rect, (sequence.size) if bounded else None))

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_UnionScore(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
    else:
        p_standard = 1

    r = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    p = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    p = 1 - np.power(p - p_standard, 2)

    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = r * p * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0

def compute_Norm_OSLE(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    gt_rect = []
    for gt_poly in sequence.groundtruth():
        if len(gt_rect) == 0:
            gt_rect.append(Special(1))
            continue
        points = np.array(gt_poly.points())
        x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
        gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))

    overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    base_overlaps = np.array(calculate_overlaps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None))
    base_overlaps[0] += 0.00001
    overlaps = overlaps / base_overlaps
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_OSLE_plot(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True):
    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
    else:
        gt_rect = sequence.groundtruth()

    overlaps = np.array(calculate_overlaps(trajectory, gt_rect, (sequence.size) if bounded else None))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    overlaps = overlaps[mask]

    step = np.arange(0, 1.01, 0.02)
    # step = np.arange(0, 1.01, 0.01)
    output = []
    number_frame = np.sum(mask)
    for s in step:
        output.append(np.sum(overlaps > s) / (number_frame + 0.000001))

    if any(mask):
        return output, np.sum(mask)
    else:
        return [], 0

def compute_eao_partial(overlaps: List, success: List[bool], curve_length: int):
    phi = curve_length * [float(0)]
    active = curve_length * [float(0)]

    for o, success in zip(overlaps, success):

        o_array = np.array(o)

        for j in range(1, curve_length):

            if j < len(o):
                phi[j] += np.mean(o_array[1:j+1])
                active[j] += 1
            elif not success:
                phi[j] += np.sum(o_array[1:len(o)]) / (j - 1)
                active[j] += 1

    phi = [p / a if a > 0 else 0 for p, a in zip(phi, active)]
    return phi, active

def count_failures(trajectory: List[Region]) -> Tuple[int, int]:
    return len([region for region in trajectory if is_special(region, Special.FAILURE)]), len(trajectory)

@analysis_registry.register("accuracy")
class SequenceAccuracy(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Sequence accurarcy"

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_accuracy(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("OSLE")
class SequenceOSLE(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Sequence Overlap over Standard Location Error"

    def describe(self):
        return Measure("Overlap over Standard Location Error", "OSLE", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_OSLE(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("Precison")
class SequencePrecison(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "E Precison"

    def describe(self):
        return Measure("E Precison", "E_P", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_Eprecision(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("Recall")
class SequenceRecall(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Recall"

    def describe(self):
        return Measure("Recall", "Recall", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_recall(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("UnionScore")
class SequenceUnionScore(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "E UnionScore"

    def describe(self):
        return Measure("E UnionScore", "E_UnionScore", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_UnionScore(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("NormOSLE")
class SequenceNormOSLE(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Normalized Sequence Overlap over Standard Location Error"

    def describe(self):
        return Measure("Normalized Overlap over Standard Location Error", "NormOSLE", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_Norm_OSLE(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("OSLEplot")
class SequenceOSLEplot(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Normalized Sequence Overlap over Standard Location Error plot"

    def describe(self):
        return Measure("Normalized Overlap over Standard Location Error plot", "NormOSLEplot", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]):

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = []
        for trajectory in trajectories:
            accuracy, _ = compute_OSLE_plot(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative.append(accuracy)
        # cummulative = np.array(cummulative)
        # cummulative = np.mean(cummulative, axis=0)
        return cummulative[0]

        # return cummulative / len(trajectories),

@analysis_registry.register("average_accuracy")
class AverageAccuracy(SequenceAggregator):

    analysis = Include(SequenceAccuracy)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average accurarcy"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Accuracy", "AUC", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_OSLE")
class AverageOSLE(SequenceAggregator):

    analysis = Include(SequenceOSLE)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Maximum Average Overlap over Standard Location Error"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Maximum Average Overlap over Standard Location Error", "MA_OSLE", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_EPrecison")
class AverageEPrecison(SequenceAggregator):

    analysis = Include(SequencePrecison)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average EPrecison"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Average EPrecison", "A_EP", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_ERecall")
class AverageERecall(SequenceAggregator):

    analysis = Include(SequenceRecall)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average Recall"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Average Recall", "A_Re", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_EUnionScore")
class AverageEUnionScore(SequenceAggregator):

    analysis = Include(SequenceUnionScore)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Average UnionScore"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Average UnionScore", "A_UnionScore", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_NormOSLE")
class AverageNormOSLE(SequenceAggregator):

    analysis = Include(SequenceNormOSLE)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Normalized Maximum Average Overlap over Standard Location Error"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Normalized Maximum Average Overlap over Standard Location Error", "MA_NormOSLE", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_OSLEplot")
class AverageNormOSLEplot(SequenceAggregator):

    analysis = Include(SequenceOSLEplot)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Normalized Maximum Average Overlap over Standard Location Error plot"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("Normalized Maximum Average Overlap over Standard Location Error plot", "MA_NormOSLEplot", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        result_grid = np.array(results)
        accuracy = np.mean(result_grid, axis=0)
        return np.sum(accuracy) * 0.01,

# -------------------------------------------------

def compute_EIoU(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
    else:
        gt_rect = sequence.groundtruth()

    # O
    overlaps = np.array(calculate_overlaps(trajectory, gt_rect, (sequence.size) if bounded else None))
    # D
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    # OD
    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_EIoUplot(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True):
    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
    else:
        gt_rect = sequence.groundtruth()

    overlaps = np.array(calculate_overlaps(trajectory, gt_rect, (sequence.size) if bounded else None))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    overlaps = overlaps[mask]

    step = np.arange(0, 1.01, 0.02)
    # step = np.arange(0, 1.01, 0.01)
    output = []
    number_frame = np.sum(mask)
    for s in step:
        output.append(np.sum(overlaps > s) / (number_frame + 0.000001))

    if any(mask):
        return output, np.sum(mask)
    else:
        return [], 0


@analysis_registry.register("EIoU")
class SequenceEIoU(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "EIoU"

    def describe(self):
        return Measure("EIoU", "EIoU", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_EIoU(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("EIoUplot")
class SequenceEIoUplot(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "EIoUplot"

    def describe(self):
        return Measure("EIoUplot", "EIoUplot", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]):

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = []
        for trajectory in trajectories:
            accuracy, _ = compute_EIoUplot(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative.append(accuracy)
        # cummulative = np.array(cummulative)
        # cummulative = np.mean(cummulative, axis=0)
        return cummulative[0]

        # return cummulative / len(trajectories),

@analysis_registry.register("average_EIoU")
class AverageEIoU(SequenceAggregator):

    analysis = Include(SequenceEIoU)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_EIoU"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_EIoU", "average_EIoU", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_EIoUplot")
class AverageEIoUplot(SequenceAggregator):

    analysis = Include(SequenceEIoUplot)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_EIoUplot"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_EIoUplot", "average_EIoUplot", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,


def compute_ENUS05(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
        p_standard[0] = 1
    else:
        p_standard = 1
    #print(p_standard)

    P = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    R = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    overlaps = R * (1 - np.power(np.abs(P/p_standard - 1), 0.5))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_ENUS1(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
        p_standard[0] = 1

    else:
        p_standard = 1
    #print(p_standard)
    P = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    R = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    overlaps = R * (1 - np.power(np.abs(P/p_standard - 1), 1))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_ENUS2(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True) -> float:

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
        p_standard[0] = 1

    else:
        p_standard = 1
    #print(p_standard)
    P = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    R = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    overlaps = R * (1 - np.power(np.abs(P/p_standard - 1), 2))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    if any(mask):
        return np.mean(overlaps[mask]), np.sum(mask)
    else:
        return 0, 0


def compute_ENUS1plot(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True):

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))
        p_standard = calculate_ps(gt_rect, sequence.groundtruth(), (sequence.size) if bounded else None)
        p_standard[0] = 1

    else:
        p_standard = 1
    #print(p_standard)
    P = np.array(calculate_ps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    R = np.array(calculate_rs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    overlaps = R * (1 - np.power(np.abs(P/p_standard - 1), 1))
    location_errs = np.array(calculate_location_errs(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))

    overlaps = overlaps * location_errs

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    overlaps = overlaps[mask]

    # step = np.arange(0, 1.01, 0.02)
    step = np.arange(0, 1.01, 0.01)
    output = []
    number_frame = np.sum(mask)
    for s in step:
        output.append(np.sum(overlaps > s) / (number_frame + 0.000001))

    if any(mask):
        return output, np.sum(mask)
    else:
        return [], 0

def compute_Recallplot(trajectory: List[Region], sequence: Sequence, burnin: int = 1,
    ignore_unknown: bool = True, bounded: bool = True):

    if isinstance(trajectory[1], Rectangle):
        gt_rect = []
        for gt_poly in sequence.groundtruth():
            if len(gt_rect) == 0:
                gt_rect.append(Special(1))
                continue
            points = np.array(gt_poly.points())
            x1, y1, x2, y2 = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            gt_rect.append(Rectangle(x1, y1, x2 - x1, y2 - y1))

    else:
        gt_rect = sequence.groundtruth()
    #print(p_standard)
    R = np.array(calculate_rs(trajectory, gt_rect, (sequence.size) if bounded else None))
    overlaps = R

    mask = np.ones(len(overlaps), dtype=bool)

    for i, region in enumerate(trajectory):
        if is_special(region, Special.UNKNOWN) and ignore_unknown:
            mask[i] = False
        elif is_special(region, Special.INITIALIZATION):
            for j in range(i, min(len(trajectory), i + burnin)):
                mask[j] = False
        elif is_special(region, Special.FAILURE):
            mask[i] = False

    overlaps = overlaps[mask]

    # step = np.arange(0, 1.01, 0.02)
    step = np.arange(0, 1.01, 0.01)
    output = []
    number_frame = np.sum(mask)
    for s in step:
        output.append(np.sum(overlaps > s) / (number_frame + 0.000001))

    if any(mask):
        return output, np.sum(mask)
    else:
        return [], 0


@analysis_registry.register("ENUS05")
class SequenceENUS05(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "ENUS05"

    def describe(self):
        return Measure("ENUS05", "ENUS05", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_ENUS05(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("ENUS1")
class SequenceENUS1(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "ENUS1"

    def describe(self):
        return Measure("ENUS1", "ENUS1", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_ENUS1(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("ENUS2")
class SequenceENUS2(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "ENUS2"

    def describe(self):
        return Measure("ENUS2", "ENUS2", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = 0
        for trajectory in trajectories:
            accuracy, _ = compute_ENUS2(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative = cummulative + accuracy

        return cummulative / len(trajectories),

@analysis_registry.register("ENUS1plot")
class SequenceENUS1plot(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "ENUS1plot"

    def describe(self):
        return Measure("ENUS1plot", "ENUS1plot", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]):

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = []
        for trajectory in trajectories:
            accuracy, _ = compute_ENUS1plot(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative.append(accuracy)
        # cummulative = np.array(cummulative)
        # cummulative = np.mean(cummulative, axis=0)
        return cummulative[0]

        # return cummulative / len(trajectories),

@analysis_registry.register("Recallplot")
class SequenceRecallplot(SeparableAnalysis):

    burnin = Integer(default=1, val_min=0)
    ignore_unknown = Boolean(default=True)
    bounded = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "Recallplot"

    def describe(self):
        return Measure("Recallplot", "Recallplot", 0, 1, Sorting.DESCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]):

        assert isinstance(experiment, MultiRunExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        cummulative = []
        for trajectory in trajectories:
            accuracy, _ = compute_Recallplot(trajectory.regions(), sequence, self.burnin, self.ignore_unknown, self.bounded)
            cummulative.append(accuracy)
        # cummulative = np.array(cummulative)
        # cummulative = np.mean(cummulative, axis=0)
        return cummulative[0]

        # return cummulative / len(trajectories),

@analysis_registry.register("average_ENUS05")
class AverageENUS05(SequenceAggregator):

    analysis = Include(SequenceENUS05)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_ENUS05"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_ENUS05", "average_ENUS05", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_ENUS1")
class AverageENUS1(SequenceAggregator):

    analysis = Include(SequenceENUS1)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_ENUS1"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_ENUS1", "average_ENUS1", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_ENUS1plot")
class AverageENUS1plot(SequenceAggregator):

    analysis = Include(SequenceENUS1plot)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_ENUS1plot"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_ENUS1plot", "average_ENUS1plot", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_ENUS2")
class AverageENUS2(SequenceAggregator):

    analysis = Include(SequenceENUS2)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_ENUS2"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_ENUS2", "average_ENUS2", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,

@analysis_registry.register("average_Recall")
class AverageRecall(SequenceAggregator):

    analysis = Include(SequenceRecall)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_Recall"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_Recall", "average_Recall", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,


@analysis_registry.register("average_Recallplot")
class AverageRecallplot(SequenceAggregator):

    analysis = Include(SequenceRecallplot)
    weighted = Boolean(default=True)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, MultiRunExperiment)

    @property
    def title(self):
        return "average_Recallplot"

    def dependencies(self):
        return self.analysis,

    def describe(self):
        return Measure("average_Recallplot", "average_Recallplot", 0, 1, Sorting.DESCENDING),

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        accuracy = 0
        frames = 0

        for i, sequence in enumerate(sequences):
            if results[i, 0] is None:
                continue

            if self.weighted:
                accuracy += results[i, 0][0] * len(sequence)
                frames += len(sequence)
            else:
                accuracy += results[i, 0][0]
                frames += 1
        return accuracy / frames,


# -------------------------------------------------

@analysis_registry.register("failures")
class FailureCount(SeparableAnalysis):

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    @property
    def title(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING),

    def subcompute(self, experiment: Experiment, tracker: Tracker, sequence: Sequence, dependencies: List[Grid]) -> Tuple[Any]:

        assert isinstance(experiment, SupervisedExperiment)

        trajectories = experiment.gather(tracker, sequence)

        if len(trajectories) == 0:
            raise MissingResultsException()

        failures = 0
        for trajectory in trajectories:
            failures = failures + count_failures(trajectory.regions())[0]

        return failures / len(trajectories), len(trajectories[0])


@analysis_registry.register("cumulative_failures")
class CumulativeFailureCount(SequenceAggregator):

    analysis = Include(FailureCount)

    def compatible(self, experiment: Experiment):
        return isinstance(experiment, SupervisedExperiment)

    def dependencies(self):
        return self.analysis,

    @property
    def title(self):
        return "Number of failures"

    def describe(self):
        return Measure("Failures", "F", 0, None, Sorting.ASCENDING), 

    def aggregate(self, _: Tracker, sequences: List[Sequence], results: Grid):
        failures = 0

        for a in results:
            failures = failures + a[0]

        return failures,
