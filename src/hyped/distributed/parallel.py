"""Parallel Data Pipe."""
import operator
from copy import deepcopy
from functools import reduce
from typing import Any

import datasets
import ray
from hyped.common.feature_checks import check_feature_equals
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor
from ray.actor import ActorHandle

import hyped.distributed.pool

from .pipe import DistributedDataPipe, RemoteDataPipe
from .pool import ActorPool


class ParallelDataPipeMixin(object):
    """Parallel Data Pipe Mixin.

    Overwrite some functions of the `DataPipe` interface according
    to the parallel workflow setup.
    """

    def prepare(self, features: datasets.Features) -> datasets.Features:
        """Prepare all data processors of the data pipe for execution.

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features of the output of the processor
        """
        # save a copy of the input features
        self._in_features = deepcopy(features)
        # prepare the data processor
        return reduce(operator.or_, (pipe.prepare(features) for pipe in self))

    @property
    def is_prepared(self) -> bool:
        """Check if all data pipes are prepared and ready for execution."""
        return all(
            (
                pipe.is_prepared
                and check_feature_equals(self.in_features, pipe.in_features)
            )
            for pipe in self
        )

    @property
    def in_features(self) -> datasets.Features:
        """Input dataset features."""
        return self._in_features

    @property
    def new_features(self) -> datasets.Features:
        """New dataset features generated by data pipe."""
        return datasets.Features(
            reduce(operator.or_, (pipe.new_features for pipe in self))
        )

    @property
    def out_features(self) -> datasets.Features:
        """All output features of the processor.

        Includes both input features and new features generated by the data pipe.
        On conflicts, the new features are prioritized.
        """
        return datasets.Features(
            reduce(operator.or_, (pipe.out_features for pipe in self))
        )

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> dict[str, list[Any]]:
        """Process a batch of examples.

        Sends the batch of examples to each distributed data pipe
        to be processed in parallel and merges the results after.

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank
            return_index (bool):
                whether to return the source index for each output example

        Returns:
            out (dict[str, list[Any]]): processed examples
            idx (list[int]):
                the source index of each processed example, only returned
                when `return_index` is set
        """
        # make sure all pools are read
        if any(not pipe.is_pool_ready for pipe in self):
            raise RuntimeError(
                "Actor Pool of `DistributedDataPipe` not initialized. "
                "This might occur when a standard a `DistributedDataPipe` "
                "is nested in a standard `DataPipe`."
            )

        futures = [None] * len(self)
        output_batches = [None] * len(self)

        reserved_actors = []
        # reserve an actor from each distributed pipe
        for i, actor in ActorPool.reserve_from_each(
            [pipe._pool for pipe in self]
        ):
            reserved_actors.append(actor)
            # schedule workload on remote actor
            futures[i] = actor.actor.batch_process.remote(
                examples=examples,
                index=index,
                return_index=True,
            )

        future2idx = {f: i for i, f in enumerate(futures)}
        # collect all outputs
        while any(out_batch is None for out_batch in output_batches):
            # wait for an actor to finish
            dones, futures = ray.wait(futures, num_returns=1)
            dones_idx = list(map(future2idx.get, dones))
            assert all(idx is not None for idx in dones_idx)

            # release actors
            for done_idx in dones_idx:
                reserved_actors[done_idx].release()
                reserved_actors[done_idx] = None

            # collect outputs from done actors
            for idx, (out_batch, out_index) in zip(dones_idx, ray.get(dones)):
                # augmentation or filtering is not supported parallel
                # data pipes
                if len(set(out_index)) != len(index):
                    raise RuntimeError(
                        "Parallel Data Pipes do not support augmentation or "
                        "filtering of examples in their sub-pipes. Detected "
                        "change in batch size after applying pipe %s at "
                        "index %i. Got %i != %i"
                        % (self[idx], idx, len(set(out_index)), len(index))
                    )

                # reorder the batch to the original order
                if any(i != j for i, j in zip(index, out_index)):
                    raise NotImplementedError()

                output_batches[idx] = out_batch

        # make sure all reserved actors are released
        assert all(actor is None for actor in reserved_actors)
        # get all outputs and merge them
        merged_out_batch = reduce(operator.or_, output_batches)

        return (merged_out_batch, index) if return_index else merged_out_batch


class RemoteParallelDataPipe(ParallelDataPipeMixin, RemoteDataPipe):
    """(Internal) Remote Parallel Data Pipe."""


class DistributedParallelDataPipe(ParallelDataPipeMixin, DistributedDataPipe):
    """Distributed Parallel Data Pipe.

    Executes the list of distributed data pipes in parallel and
    merges their outputs after all pipes have finished. The merging
    is done in the order of the pipes in the input list. In case
    of conflicting outputs, the one of the data pipe lastest in the
    list is kept.

    Arguments:
        processors (list[DataPipe | DistributedDataPipe | BaseDataProcessor]):
            data processors to be executed in parallel. Note that these
            can also be data pipes in case one wants to parallelize not
            just a single processor.
        num_proc_per_pipe (None | int):
            number of workers to spawn per data pipe. By default this value
            is taken from the `num_proc` argument to the `.apply` function.
    """

    def __init__(
        self,
        processors: list[DataPipe | DistributedDataPipe | BaseDataProcessor],
        num_proc_per_pipe: None | int = None,
    ) -> None:
        """Constructor."""
        # convert all processors to distributed data pipes
        processors = [
            proc
            if isinstance(proc, DistributedDataPipe)
            else DistributedDataPipe(list(proc))
            if isinstance(proc, DataPipe)
            else DistributedDataPipe([proc])
            for proc in processors
        ]
        # initialize as a standard data pipe
        super(DistributedParallelDataPipe, self).__init__(
            processors=processors, num_proc=num_proc_per_pipe
        )

    def _spawn_actor(self, rank: int) -> ActorHandle:
        """Spawn a single remote worker actor."""
        return ray.remote(RemoteParallelDataPipe).remote(list(self), rank)

    def prepare(self, features: datasets.Features) -> datasets.Features:
        """Prepare all data pipes for execution.

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features after applying all data pipes
        """
        # prepare main process
        out_features = ParallelDataPipeMixin.prepare(self, features)

        if (hyped.distributed.pool.rank is None) or (
            hyped.distributed.pool.rank == 0
        ):
            with self._pool.reserve_all() as reserved_actors:
                # make sure all actors are reserved for preparation
                assert len(reserved_actors) == self.num_proc

                # prepare all actors
                for actor_out_features in ray.get(
                    reserved_actors.for_all_actors(
                        lambda a: a.prepare.remote(features)
                    )
                ):
                    assert check_feature_equals(
                        actor_out_features, out_features
                    )

        return out_features
