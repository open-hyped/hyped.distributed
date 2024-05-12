"""Distributed Data Pipe."""
from __future__ import annotations

import traceback
from typing import Any, Iterable

import datasets
import ray
from hyped.common.feature_checks import check_feature_equals
from hyped.data.pipe import DataPipe, DatasetType
from hyped.data.processors.base import BaseDataProcessor
from ray.actor import ActorHandle
from ray.util.queue import Empty, Queue

import hyped.distributed.pool

from .map import (
    _map_dataset,
    _map_dataset_dict,
    _map_iterable_dataset,
    _map_iterable_dataset_dict,
)
from .pool import ActorPool, RemoteWorker


class RemoteDataPipe(DataPipe, RemoteWorker):
    """(Internal) Remote Data Pipe.

    Class that is distributed internally by `DistributedDataPipe`.
    Provides helper functions for distributed setting on top of
    the standard `DataPipe` functionality.

    Arguments:
        processors (list[BaseDataProcessor | DataPipe]):
            pipe of processors
        rank (int): rank of the remote data pipe
    """

    def __init__(
        self, processors: list[BaseDataProcessor, DataPipe] = [], rank: int = 0
    ) -> None:
        """Constructor."""
        RemoteWorker.__init__(self, rank)
        DataPipe.__init__(self, processors)

    def _self(self, attr_name: str) -> Any:
        return getattr(self, attr_name)

    def _set_pool(self, idx: int | tuple[int], pool: ActorPool) -> None:
        pipe = self._at_idx(idx)
        # make sure the processor at the given index
        # is a distributed data pipe
        if not isinstance(pipe, DistributedDataPipe):
            raise TypeError(
                "Expected `DistributedDataPipe` instance at index "
                "%s, got %s" % (str(idx), pipe)
            )
        pipe._set_pool(pool)

    def _at_idx(self, idx: int | tuple[int]) -> ActorPool:
        idx = (idx,) if isinstance(idx, int) else idx

        p = self

        for j, i in enumerate(idx):
            # make sure the processor at the given index
            # is a distributed data pipe
            if not isinstance(p, DataPipe):
                raise TypeError(
                    "Expected `DataPipe` instance at index "
                    "%s, got %s" % (str(idx[:j]), p)
                )
            p = p[i]

        return p

    def _map_single(
        self, shard: datasets.Dataset, update_queue: Queue, **kwargs
    ) -> None:
        # doesn't accept the rank argument
        # rank is inferred from the worker instance
        if "rank" in kwargs:
            raise TypeError(
                "`_map_single` got an unexpected keyword argument `rank`"
            )

        # hard code some keyword arguments
        kwargs |= dict(
            batched=True,
            with_indices=True,
            with_rank=True,
            features=self.out_features,
            rank=self.get_rank(),
        )

        try:
            for content in datasets.Dataset._map_single(
                shard=shard, function=self._batch_process_to_pyarrow, **kwargs
            ):
                update_queue.put(content)
        except Exception as e:
            tb = "".join(traceback.format_exception(e))
            update_queue.put((self.get_rank(), False, Exception(tb)))


# TODO: remote data pipes currently do not support statistics
#       processors, this requires the statistics to be send
#       from the remote actor to the main process
class DistributedDataPipe(DataPipe):
    """Distributed Data Pipe.

    Uses `ray` to distribute workload over a ray cluster. It
    creates a number of Actors of the `RemoteDataPipe` type to
    which the data is being distributed during processing.

    Arguments:
        processors (list[BaseDataProcessor | DataPipe]):
            the initial pipe of processors
        num_proc (None | int):
            number of distributed workers to spawn. By default this
            value is taken from the `num_proc` argument to the
            `DistributedDataPipe.apply` function. However, this can
            be set explicitly to allow different degrees of
            parallelism for different components of the data pipe.
        proc_options (dict[str, Any]):
            arguments forwarded to `ray.remote` function, used to
            specify the required resources per process. For more
            infomation please refer to the ray documentation.
    """

    def __init__(
        self,
        processors: list[BaseDataProcessor, DataPipe] = [],
        num_proc: None | int = None,
        proc_options: dict[str, Any] = {},
    ) -> None:
        """Constructor."""
        super(DistributedDataPipe, self).__init__(processors)

        self._options = proc_options
        # pool of all worker actors
        self._pool: None | ActorPool = None
        # spawn all actors if number of processes is specified
        if num_proc is not None:
            self._spawn_pool(num_actors=num_proc)

    def _set_pool(self, pool: ActorPool) -> None:
        """Set the worker pool.

        Arguments:
            pool (ActorPool): actor pool
        """
        assert not self.is_pool_ready
        self._pool = pool

    def _spawn_actor(self, rank: int) -> ActorHandle:
        """Spawn a single remote worker actor."""
        return (
            ray.remote(**self._options)
            if len(self._options) > 0
            else ray.remote
        )(RemoteDataPipe).remote(list(self), rank)

    def _spawn_pool(self, num_actors: int) -> ActorPool:
        """Spawn remote actors.

        Spawn all remote actors of the distributed data pipe
        including nested distributed data pipes.

        Arguments:
            num_actors (int): number of actors to spawn

        Returns:
            pool (ActorPool): actor pool of distributed data pipe
        """

        def _spawn_nested_pools(pipe: DataPipe) -> dict[int, ActorPool]:
            nested_pools = {}
            # spawn pool in nested data pipes
            for i, p in enumerate(pipe):
                if isinstance(p, DistributedDataPipe) and not p.is_pool_ready:
                    # spawn the actor pool of the nested data pipe
                    # use as many actors as the parent data pipe
                    nested_pools[(i,)] = p._spawn_pool(num_actors=num_actors)
                elif isinstance(p, DataPipe) and not isinstance(
                    p, DistributedDataPipe
                ):
                    # look for distributed data pipes
                    # nested in standard data pipes
                    for nested_idx, pool in _spawn_nested_pools(p).items():
                        nested_pools[(i,) + nested_idx] = pool
            return nested_pools

        nested_pools = _spawn_nested_pools(self)

        # set actor pool for distributed data pipe
        self._set_pool(
            ActorPool([self._spawn_actor(rank) for rank in range(num_actors)])
        )

        # reserve all actors in the pool
        with self._pool.reserve_all() as reserved_actors:
            # set the actor pools of nested pipes in remote actors
            for idx, pool in nested_pools.items():
                ray.wait(
                    reserved_actors.for_all_actors(
                        lambda a: a._set_pool.remote(idx, pool)
                    ),
                    num_returns=len(reserved_actors),
                )

        return self._pool

    @property
    def is_pool_ready(self) -> bool:
        """Checks whether the actor pool is ready."""
        return self._pool is not None

    @property
    def num_proc(self) -> int | None:
        """Number of distributed workers/processes used.

        Returns None if the actors are not ready.
        """
        return self._pool.num_actors if self.is_pool_ready else None

    @property
    def is_prepared(self) -> bool:
        """Check if the data pipe is prepared and ready for execution.

        This check includes the check that the data pipe is prepared,
        the worker pool is initialized and ready for usage and all
        workers in the pool are prepared.
        """
        is_prepared = (
            super(DistributedDataPipe, self).is_prepared and self.is_pool_ready
        )

        if is_prepared and (
            (hyped.distributed.pool.rank is None)
            or (hyped.distributed.pool.rank == 0)
        ):
            # make sure all workers are prepared as well
            is_prepared = is_prepared and all(
                ray.get(
                    [
                        actor._self.remote("is_prepared")
                        for actor in self._pool.actors.values()
                    ]
                )
            )

        return is_prepared

    def _check_actor(self, actor: ActorHandle) -> None:
        """Check configuration of the given actor."""
        assert super(DistributedDataPipe, self).is_prepared == ray.get(
            actor._self.remote("is_prepared")
        )

        if super(DistributedDataPipe, self).is_prepared:
            for feature_name in [
                "in_features",
                "new_features",
                "out_features",
            ]:
                target = getattr(self, feature_name)
                query = ray.get(actor._self.remote(feature_name))
                assert check_feature_equals(query, target)

    def prepare(self, features: datasets.Features) -> datasets.Features:
        """Prepare the distributed data pipe.

        This includes the preparation of the main data pipe as well as
        all remote data pipes.

        Arguments:
            features (datasets.Features): input features

        Return:
            out_features (datasets.Features): output features
        """
        # prepare main process data pipe
        out_features = super(DistributedDataPipe, self).prepare(features)
        assert super(DistributedDataPipe, self).is_prepared

        if not self.is_pool_ready:
            raise RuntimeError(
                "Actor pool not initialized. Please make sure the "
                "pool is ready before calling `prepare`."
            )

        # let only rank zero start the preparation
        # of remote data pipes
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

            # check actors
            for actor in reserved_actors.actors:
                self._check_actor(actor)

        return out_features

    def batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> dict[str, list[Any]]:
        """Process a batch of examples.

        Sends the batch of examples to the a remote data pipe. The selection
        strategy of which remote data pipe to send the data to is implemented
        by the `_get_actor` function.

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
        if not self.is_pool_ready:
            raise RuntimeError(
                "Actor Pool of `DistributedDataPipe` not initialized. "
                "This might occur when a standard a `DistributedDataPipe` "
                "is nested in a standard `DataPipe`. Fix this error by "
                "specifying the `num_proc` argument in the constructor of"
                "the distributed data pipe."
            )

        with self._pool.reserve() as (_, actor):
            # call function on actor and get output
            output = ray.get(
                actor.batch_process.remote(
                    examples=examples,
                    index=index,
                    return_index=return_index,
                )
            )

        return output

    def iter_batch_process(
        self,
        examples: dict[str, list[Any]],
        index: list[int],
        rank: None | int = None,
        return_index: bool = False,
    ) -> Iterable[
        dict[str, list[Any]] | tuple[dict[str, list[Any]], list[int]]
    ]:
        """Apply each data processor to the batch of examples.

        Yields the output of each data processor when ready.

        Arguments:
            examples (dict[str, list[Any]]):
                batch of examples to process
            index (list[int]):
                dataset indices of the examples
            rank (int):
                execution process rank
            return_index (bool):
                whether to return the source index for each output example


        Returns:
            out_iter (
                Iterable[dict[str, list[Any]]
                | tuple[dict[str, list[Any]], list[int]]]
            ):
                iterator over output batch from each processor
        """
        raise NotImplementedError()

    def internal_apply(
        self,
        data: DatasetType,
        **kwargs,
    ) -> DatasetType:
        """Internal apply function.

        Applies the distributed implementations of the dataset map
        functions (see `hyped.distributed.map`).

        Arguments:
            data (DatasetType):
                source dataset(s)
            **kwargs:
                keyword arguments passed to the map function
                appropriate for the given dataset type

        Returns:
            out (DatasetType):
                processed dataset(s)
        """
        if isinstance(data, datasets.Dataset):
            return _map_dataset(self=data, pipe=self, **kwargs)

        if isinstance(data, datasets.DatasetDict):
            return _map_dataset_dict(self=data, pipe=self, **kwargs)

        if isinstance(data, datasets.IterableDataset):
            return _map_iterable_dataset(self=data, pipe=self, **kwargs)

        if isinstance(data, datasets.IterableDatasetDict):
            return _map_iterable_dataset_dict(self=data, pipe=self, **kwargs)

        raise ValueError(
            "`DistributedDataPipe` got an invalid dataset type, got %s "
            "of type %s" % (str(data), type(data))
        )

    def apply(
        self,
        data: DatasetType,
        **kwargs,
    ) -> DatasetType:
        """Apply the data pipe to a dataset.

        Arguments:
            data (Dataset|DatasetDict|IterableDataset|IterableDatasetDict):
                source dataset(s)
            **kwargs (dict[str, Any]):
                arguments forwarded to the `map` function used
                for the specific dataset type

        Returns:
            out (datasets.Dataset|datasets.DatasetDict): processed dataset(s)
        """
        # get the number of processes specified
        # in the arguments
        num_proc = kwargs.pop("num_proc", None)

        # check the number of processes argument
        if (
            self.is_pool_ready
            and (num_proc is not None)
            and (num_proc != self.num_proc)
        ):
            raise ValueError(
                "Got ambiguous values for `num_proc` argument. "
                "Please provide the argument either in the "
                "constructor or the `apply` function, but not both."
                "Got %i != %i" % (self.num_proc, num_proc)
            )

        elif not self.is_pool_ready:
            # spawn remote actors
            self._spawn_pool(
                num_actors=num_proc if num_proc is not None else 1
            )
            assert self.is_pool_ready

        return super(DistributedDataPipe, self).apply(data, **kwargs)
