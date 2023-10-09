#pylint: disable=W0223

import numpy as np

from typing import Callable

from vot.dataset import Sequence
from vot.region import Special, calculate_overlap
from vot.region.shapes import Shape, Rectangle, Polygon, Mask

from attributee import Boolean, Integer, Float, List, String

from vot.experiment import Experiment, experiment_registry
from vot.tracker import Tracker, Trajectory
from copy import deepcopy


class MultiRunExperiment(Experiment):

    repetitions = Integer(val_min=1, default=1)
    shake_rate = Float(val_min=0, default=0.8)
    early_stop = Boolean(default=True)

    def _can_stop(self, tracker: Tracker, sequence: Sequence):
        if not self.early_stop:
            return False
        trajectories = self.gather(tracker, sequence)
        if len(trajectories) < 3:
            return False

        for trajectory in trajectories[1:]:
            if not trajectory.equals(trajectories[0]):
                return False

        return True

    def scan(self, tracker: Tracker, sequence: Sequence):
        
        results = self.results(tracker, sequence)

        files = []
        complete = True

        for i in range(1, self.repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                files.extend(Trajectory.gather(results, name))
            elif self._can_stop(tracker, sequence):
                break
            else:
                complete = False
                break

        return complete, files, results

    def gather(self, tracker: Tracker, sequence: Sequence):
        trajectories = list()
        results = self.results(tracker, sequence)
        for i in range(1, self.repetitions+1):
            name = "%s_%03d" % (sequence.name, i)
            if Trajectory.exists(results, name):
                trajectories.append(Trajectory.read(results, name))
        return trajectories

@experiment_registry.register("unsupervised_shake")
class UnsupervisedShakeExperiment(MultiRunExperiment):

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            assert self.repetitions % 6 == 0, 'Unsupervised ShakeRepetitions number error!'

            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)
                initial_shape = deepcopy(self._get_initialization(sequence, 0))
                i_offset = (i-1) % 6 + 1

                if isinstance(initial_shape, Rectangle):
                    data1 = np.round(initial_shape._data)
                    x1, y1, w, h = data1[:, 0]
                    x2, y2 = x1 + w, y1 + h
                    w, h = x2 - x1, y2 - y1
                    shake_w, shake_h = self.shake_rate * w, self.shake_rate * h
                    if i_offset == 1:
                        x1 -= shake_w
                        y1 -= shake_h
                    elif i_offset == 2:
                        x2 += shake_w
                        y1 -= shake_h
                    elif i_offset == 3:
                        x1 -= shake_w
                        y2 += shake_h
                    elif i_offset == 4:
                        x2 += shake_w
                        y2 += shake_h
                    elif i_offset == 5:
                        x1 -= shake_w
                        y1 -= shake_h
                        x2 += shake_w
                        y2 += shake_h
                    elif i_offset == 6:
                        x1 += shake_w
                        y1 += shake_h
                        x2 -= shake_w
                        y2 -= shake_h

                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, sequence.width)
                    y2 = min(y2, sequence.height)

                    initial_shape = Rectangle(x1, y1, x2 - x1, y2 - y1)

                elif isinstance(initial_shape, Polygon):
                    data1 = np.round(initial_shape._points)
                    x1, y1, x2, y2 = np.min(data1[:, 0]), np.min(data1[:, 1]), np.max(data1[:, 0]), np.max(data1[:, 1])

                    r = 1 + self.shake_rate

                    if i_offset == 1:
                        x0 = x2
                        y0 = y2
                    elif i_offset == 2:
                        x0 = x1
                        y0 = y2
                    elif i_offset == 3:
                        x0 = x2
                        y0 = y1
                    elif i_offset == 4:
                        x0 = x1
                        y0 = y1
                    elif i_offset == 5:
                        x0 = (x1 + x2) / 2
                        y0 = (y1 + y2) / 2
                    elif i_offset == 6:
                        x0 = (x1 + x2) / 2
                        y0 = (y1 + y2) / 2
                        r = 1 - self.shake_rate
                    initial_shape.resize_center(x0, y0, r, sequence.width, sequence.height)

                else:
                    assert False, 'Type error initial shape!'

                _, properties, elapsed = runtime.initialize(sequence.frame(0), initial_shape)

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, sequence.length):
                    region, properties, elapsed = runtime.update(sequence.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

                trajectory.write(results, name)

                if callback:
                    callback(i / self.repetitions)

@experiment_registry.register("unsupervised")
class UnsupervisedExperiment(MultiRunExperiment):

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            for i in range(1, self.repetitions+1):
                # for
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)

                _, properties, elapsed = runtime.initialize(sequence.frame(0), self._get_initialization(sequence, 0))

                properties["time"] = elapsed

                trajectory.set(0, Special(Special.INITIALIZATION), properties)

                for frame in range(1, sequence.length):
                    region, properties, elapsed = runtime.update(sequence.frame(frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, region, properties)

                trajectory.write(results, name)

                if callback:
                    callback(i / self.repetitions)

@experiment_registry.register("supervised")
class SupervisedExperiment(MultiRunExperiment):

    skip_initialize = Integer(val_min=1, default=1)
    skip_tags = List(String(), default=[])
    failure_overlap = Float(val_min=0, val_max=1, default=0)

    def execute(self, tracker: Tracker, sequence: Sequence, force: bool = False, callback: Callable = None):

        results = self.results(tracker, sequence)

        with self._get_runtime(tracker, sequence) as runtime:

            for i in range(1, self.repetitions+1):
                name = "%s_%03d" % (sequence.name, i)

                if Trajectory.exists(results, name) and not force:
                    continue

                if self._can_stop(tracker, sequence):
                    return

                trajectory = Trajectory(sequence.length)

                frame = 0
                while frame < sequence.length:

                    _, properties, elapsed = runtime.initialize(sequence.frame(frame), self._get_initialization(sequence, frame))

                    properties["time"] = elapsed

                    trajectory.set(frame, Special(Special.INITIALIZATION), properties)

                    frame = frame + 1

                    while frame < sequence.length:

                        region, properties, elapsed = runtime.update(sequence.frame(frame))

                        properties["time"] = elapsed

                        if calculate_overlap(region, sequence.groundtruth(frame), sequence.size) <= self.failure_overlap:
                            trajectory.set(frame, Special(Special.FAILURE), properties)
                            frame = frame + self.skip_initialize
 
                            if self.skip_tags:
                                while frame < sequence.length:
                                    if not [t for t in sequence.tags(frame) if t in self.skip_tags]:
                                        break
                                    frame = frame + 1
                            break
                        else:
                            trajectory.set(frame, region, properties)
                        frame = frame + 1

                if  callback:
                    callback(i / self.repetitions)

                trajectory.write(results, name)
