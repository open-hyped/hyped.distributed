"""Distributed dataset map functionality."""
from __future__ import annotations

import os
from copy import deepcopy
from itertools import compress, count, islice
from math import ceil
from typing import Callable, Dict, Iterator, Literal, Optional, TypeVar, Union

import datasets
import numpy as np
import pyarrow as pa
import ray
from datasets.arrow_dataset import (
    NonExistentDatasetError,
    _concatenate_map_style_datasets,
    transmit_format,
    transmit_tasks,
)
from datasets.fingerprint import (
    format_kwargs_for_fingerprint,
    format_transform_for_fingerprint,
    is_caching_enabled,
    update_fingerprint,
    validate_fingerprint,
)
from datasets.formatting import TensorFormatter, get_formatter
from datasets.iterable_dataset import (
    ArrowExamplesIterable,
    FormattingConfig,
    Key,
    TypedExamplesIterable,
    _BaseExamplesIterable,
    _batch_arrow_tables,
    _batch_to_examples,
    _convert_to_arrow,
    _examples_to_batch,
)
from datasets.utils import tqdm as hf_tqdm
from ray.util.queue import Queue


@transmit_tasks
@transmit_format
def _map_dataset(
    self: datasets.Dataset,
    pipe: "DistributedDataPipe",
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_name: Optional[str] = None,
    writer_batch_size: Optional[int] = 1000,
    disable_nullable: bool = False,
    new_fingerprint: Optional[str] = None,
    suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
    desc: Optional[str] = None,
) -> datasets.Dataset:
    """Distributed dataset map function.

    Adaption of the `datasets.Dataset.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset
    using ray`.

    Arguments:
        self (datasets.Dataset): dataset
        pipe (DistributedDataPipe): data pipe
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed
        keep_in_memory (bool):
            Keep the dataset in memory instead of writing it
            to a cache file.
        load_from_cache_file (bool):
            If a cache file storing the output of the pipe
            can be identified, use it instead of recomputing.
        cache_file_name (str):
            Provide the name of a path for the cache file.
        writer_batch_size (int):
            Number of rows per write operation for the cache
            file writer.
        disable_nullable (bool):
            Disallow null values in the table.
        suffix_template (str):
            If cache_file_name is specified, then this suffix will be
            added at the end of the base name of each.
        new_fingerprint (str):
            The new fingerprint of the dataset after transform.
        desc (str):
            Meaningful description to be displayed alongside
            with the progress bar while mapping examples.

    Returns:
        transformed_dataset (datasets.Dataset):
            dataset after being passed through the data pipe

    """
    if keep_in_memory and cache_file_name is not None:
        raise ValueError(
            "Please use either `keep_in_memory` or "
            "`cache_file_name` but not both."
        )

    # If the array is empty we do nothing (but we make sure to
    # handle an empty indices mapping and remove the requested
    # columns anyway)
    if len(self) == 0:
        if self._indices is not None:  # empty indices mapping
            self = Dataset(
                self.data.slice(0, 0),
                info=self.info.copy(),
                split=self.split,
                fingerprint=new_fingerprint,
            )
        if remove_columns:
            return self.remove_columns(remove_columns)
        else:
            return self

    load_from_cache_file = (
        load_from_cache_file
        if load_from_cache_file is not None
        else is_caching_enabled()
    )

    with pipe._pool.reserve_all() as actors:
        rank2actor = {a.rank: a for a in actors}
        rank2idx = {a.rank: i for i, a in enumerate(actors)}
        num_proc = num_shards = len(actors)

        if num_proc == 0:
            raise RuntimeError(
                "No remote actors available for "
                "`DistributedDataPipe` instance"
            )

        dataset_kwargs = dict(
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            keep_in_memory=keep_in_memory,
            writer_batch_size=writer_batch_size,
            disable_nullable=disable_nullable,
        )

        if new_fingerprint is None:
            # we create a unique hash from the function,
            # current dataset file and the mapping args
            transform = format_transform_for_fingerprint(
                datasets.Dataset._map_single
            )
            kwargs_for_fingerprint = format_kwargs_for_fingerprint(
                datasets.Dataset._map_single,
                (),
                dataset_kwargs | dict(shard=self, function=pipe),
            )
            kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
            new_fingerprint = update_fingerprint(
                self._fingerprint, transform, kwargs_for_fingerprint
            )
        else:
            validate_fingerprint(new_fingerprint)

        # get cache file name
        if self.cache_files and (cache_file_name is None):
            cache_file_name = self._get_cache_file_path(new_fingerprint)

        def load_processed_shard_from_cache(shard_kwargs):
            shard = shard_kwargs["shard"]
            # Check if we've already cached this computation (indexed by a hash)
            if shard_kwargs["cache_file_name"] is not None:
                if load_from_cache_file and os.path.exists(
                    shard_kwargs["cache_file_name"]
                ):
                    info = shard.info.copy()
                    info.features = features
                    info.task_templates = None
                    return Dataset.from_file(
                        shard_kwargs["cache_file_name"],
                        info=info,
                        split=shard.split,
                    )
            raise NonExistentDatasetError

        def format_cache_file_name(
            cache_file_name: Optional[str],
            rank: Union[int, Literal["*"]],  # noqa: F722
        ) -> Optional[str]:
            if not cache_file_name:
                return cache_file_name
            sep = cache_file_name.rindex(".")
            base_name, extension = cache_file_name[:sep], cache_file_name[sep:]
            if isinstance(rank, int):
                cache_file_name = (
                    base_name
                    + suffix_template.format(rank=rank, num_proc=num_proc)
                    + extension
                )
            else:
                cache_file_name = (
                    base_name
                    + suffix_template.replace("{rank:05d}", "{rank}").format(
                        rank=rank, num_proc=num_proc
                    )
                    + extension
                )
            return cache_file_name

        def format_new_fingerprint(new_fingerprint: str, rank: int) -> str:
            new_fingerprint = new_fingerprint + suffix_template.format(
                rank=rank, num_proc=num_proc
            )
            validate_fingerprint(new_fingerprint)
            return new_fingerprint

        # create one dataset shard for each worker
        shards = [
            self.shard(
                num_shards=num_shards,
                index=index,
                contiguous=True,
                keep_in_memory=keep_in_memory,
            )
            for index in range(num_shards)
        ]

        futures = []
        update_queue = Queue()
        transformed_shards = [None] * num_shards
        # start all workers
        for actor, shard in zip(actors, shards):
            formatted_cache_file_name = format_cache_file_name(
                cache_file_name, actor.rank
            )
            formatted_new_fingerprint = format_new_fingerprint(
                new_fingerprint, actor.rank
            )

            try:
                idx = rank2idx[actor.rank]
                transformed_shards[idx] = load_processed_shard_from_cache(
                    dict(
                        shard=shard, cache_file_name=formatted_cache_file_name
                    )
                )
                # free actor
                actor.release()

            except NonExistentDatasetError:
                # start workload on actor
                futures.append(
                    actor.actor._map_single.remote(
                        shard=shard,
                        update_queue=update_queue,
                        offset=sum(map(len, shards[: actor.rank])),
                        cache_file_name=formatted_cache_file_name,
                        new_fingerprint=formatted_new_fingerprint,
                        **dataset_kwargs,
                    )
                )

        pbar_total = (
            len(self)
            if not drop_last_batch
            else (
                len(self) // num_shards // batch_size * num_shards * batch_size
            )
        )

        with hf_tqdm(
            unit=" examples",
            total=pbar_total,
            desc=(desc or "Map") + f" (num_proc={num_proc})",
        ) as pbar:
            # collect all outputs
            shards_done = 0
            while shards_done < num_shards:
                # get next value from update queue
                rank, done, content = update_queue.get()

                if isinstance(content, Exception):
                    raise RuntimeError(
                        "Error in remote worker with rank %i" % rank
                    ) from content

                if done:
                    shards_done += 1
                    idx = rank2idx[rank]
                    actor = rank2actor[rank]
                    # set content and release actor
                    transformed_shards[idx] = content
                    actor.release()
                else:
                    pbar.update(content)

    # concatenate all shards
    result = _concatenate_map_style_datasets(transformed_shards)

    # update fingerprint if the dataset changed
    if any(
        transformed_shard._fingerprint != shard._fingerprint
        for transformed_shard, shard in zip(transformed_shards, shards)
    ):
        result._fingerprint = new_fingerprint
    else:
        result._fingerprint = self._fingerprint

    # make sure all workers are finished
    ray.wait(futures, num_returns=len(futures), timeout=1)

    return result


def _map_dataset_dict(
    self: datasets.DatasetDict,
    pipe: "DistributedDataPipe",
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
    keep_in_memory: bool = False,
    load_from_cache_file: Optional[bool] = None,
    cache_file_names: Optional[Dict[str, Optional[str]]] = None,
    writer_batch_size: Optional[int] = 1000,
    disable_nullable: bool = False,
    desc: Optional[str] = None,
) -> datasets.DatasetDict:
    """Distributed dataset dict map function.

    Adaption of the `datasets.DatasetDict.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset dict
    using ray`.

    Arguments:
        self (datasets.DatasetDict): dataset dict
        pipe (DistributedDataPipe): data pipe
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed
        keep_in_memory (bool):
            Keep the dataset in memory instead of writing it
            to a cache file.
        load_from_cache_file (bool):
            If a cache file storing the output of the pipe
            can be identified, use it instead of recomputing.
        cache_file_names (dict[str, Optional[str]]):
            Provide the name of a path for the cache file.
        writer_batch_size (int):
            Number of rows per write operation for the cache
            file writer.
        disable_nullable (bool):
            Disallow null values in the table.
        desc (str):
            Meaningful description to be displayed alongside
            with the progress bar while mapping examples.

    Returns:
        transformed_dataset (datasets.Dataset):
            dataset after being passed through the data pipe

    """
    self._check_values_type()
    if cache_file_names is None:
        cache_file_names = {k: None for k in self}

    return datasets.DatasetDict(
        {
            key: _map_dataset(
                self=data,
                pipe=pipe,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[key],
                writer_batch_size=writer_batch_size,
                disable_nullable=disable_nullable,
                desc=desc,
            )
            for key, data in self.items()
        }
    )


class DistributedMappedExamplesIterable(_BaseExamplesIterable):
    """Distributed Mapped Examples Iterable."""

    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        pipe: DistributedDataPipe,
        unordered: bool,
        batch_size: int,
        drop_last_batch: bool,
        formatting: FormattingConfig,
    ) -> None:
        """Constructor."""
        if batch_size is None:
            raise ValueError("batch size not specified")

        super(DistributedMappedExamplesIterable, self).__init__()

        self.ex_iterable = ex_iterable
        self.pipe = pipe

        self.unordered = unordered
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.formatting = formatting
        # make sure the actor pool of the data pipe is ready
        assert pipe.is_pool_ready

        if self.formatting and self.formatting.format_type == "arrow":
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        """Iterator."""
        if self.formatting and self.formatting.format_type == "arrow":
            yield from ArrowExamplesIterable(self._iter_arrow, {})
        else:
            yield from self._iter()

    def _iter(self):
        """Iterator."""
        iterable = iter(self.ex_iterable)
        counter = count()

        if self.formatting:
            formatter = get_formatter(self.formatting.format_type)
            format_dict = (
                formatter.recursive_tensorize
                if isinstance(formatter, TensorFormatter)
                else cast_to_python_objects
            )
        else:
            format_dict = None

        def get_mini_batches(num_batches):
            # give each worker an initial workload
            batch = list(islice(iterable, self.batch_size * num_batches))

            # check if all batches are complete
            if (
                len(batch) < self.batch_size * num_batches
            ) and self.drop_last_batch:
                # if not then drop the last incomplete batch
                n = self.batch_size * (len(batch) // self.batch_size)
                batch = batch[:n]

            if len(batch) == 0:
                return

            keys, examples = zip(*batch)
            mini_batch_size = ceil(len(batch) / num_batches)

            for i in range(num_batches):
                # get mini batch for i-th worker
                mini_batch = examples[
                    i * mini_batch_size : (i + 1) * mini_batch_size
                ]
                num_examples = len(mini_batch)
                # prepare mini batch
                mini_batch = _examples_to_batch(mini_batch)
                mini_batch = (
                    format_dict(mini_batch) if format_dict else mini_batch
                )
                # build new key for mini batch
                key = "_".join(
                    map(
                        str,
                        keys[i * mini_batch_size : (i + 1) * mini_batch_size],
                    )
                )

                yield num_examples, key, mini_batch

        # reserve all available actors
        with self.pipe._pool.reserve_all() as actors:
            num_proc = len(actors)
            assert num_proc > 0

            futures = []
            future2key = {}
            future2actor = {}
            # schedule initial workload
            for actor, (n, new_key, mini_batch) in zip(
                actors.actors, get_mini_batches(num_batches=num_proc)
            ):
                f = actor.batch_process.remote(
                    examples=mini_batch,
                    index=[next(counter) for _ in range(n)],
                )

                futures.append(f)
                future2key[f] = new_key
                future2actor[f] = actor

            while len(futures) > 0:
                if self.unordered:
                    # collect any worker that is finished
                    dones, futures = ray.wait(futures, num_returns=1)
                else:
                    # wait for the first worker in the list to finish
                    # to preserve the original order of the dataset
                    dones, _ = ray.wait([futures.pop(0)], num_returns=1)

                # collect and yield outputs
                for done, out_batch in zip(dones, ray.get(dones)):
                    actor = future2actor.pop(done)
                    new_key = future2key.pop(done)

                    # schedule next work for actor
                    # this loop either does example one iteration
                    # over one requested batch or no iteration at
                    # all when the dataset is exhausted
                    for n, new_key, mini_batch in get_mini_batches(
                        num_batches=1
                    ):
                        f = actor.batch_process.remote(
                            examples=mini_batch,
                            index=[next(counter) for _ in range(n)],
                        )

                        futures.append(f)
                        future2key[f] = new_key
                        future2actor[f] = actor

                    # yield examples from output batch
                    for example in _batch_to_examples(out_batch):
                        yield new_key, example

    def _iter_arrow(self) -> Iterator[Tuple[Key, pa.Table]]:
        """Arrow Iterator."""
        raise NotImplementedError()

    def shuffle_data_sources(
        self, generator: np.random.Generator
    ) -> DistributedMappedExamplesIterable:
        """Shuffle data sources."""
        return DistributedMappedExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator),
            pipe=self.pipe,
            unordered=self.unordered,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch,
            formatting=self.formatting,
        )

    def shard_data_sources(
        self, worker_id: int, num_workers: int
    ) -> DistributedMappedExamplesIterable:
        """Shard data sources."""
        return DistributedMappedExamplesIterable(
            self.ex_iterable.shard_data_sources(worker_id, num_workers),
            pipe=self.pipe,
            unordered=self.unordered,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch,
            formatting=self.formatting,
        )

    @property
    def n_shards(self) -> int:
        """Number of shards."""
        return self.ex_iterable.n_shards


def _map_iterable_dataset(
    self,
    pipe: "DistributedDataPipe",
    unordered: bool = True,
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
) -> datasets.IterableDataset:
    """Distributed iterable dataset map function.

    Adaption of the `datasets.IterableDataset.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset
    using ray`.

    Arguments:
        self (datasets.Dataset): dataset
        pipe (DistributedDataPipe): data pipe
        unordered (bool):
            whether the order of the dataset should be preserved
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed

    Returns:
        transformed_dataset (datasets.IterableDataset):
            dataset after being passed through the data pipe

    """
    ex_iterable = DistributedMappedExamplesIterable(
        TypedExamplesIterable(
            self._ex_iterable,
            pipe.in_features,
            token_per_repo_id=self._token_per_repo_id,
        ),
        pipe=pipe,
        unordered=unordered,
        batch_size=batch_size,
        drop_last_batch=drop_last_batch,
        formatting=self._formatting,
    )

    info = self.info.copy()
    info.features = pipe.out_features

    return datasets.IterableDataset(
        ex_iterable=ex_iterable,
        info=info,
        split=self._split,
        formatting=self._formatting,
        shuffling=deepcopy(self._shuffling),
        distributed=deepcopy(self._distributed),
        token_per_repo_id=self._token_per_repo_id,
    )


def _map_iterable_dataset_dict(
    self,
    pipe: "DistributedDataPipe",
    unordered: bool = True,
    batch_size: Optional[int] = 1000,
    drop_last_batch: bool = False,
) -> datasets.IterableDatasetDict:
    """Distributed iterable dataset dict map function.

    Adaption of the `datasets.IterableDatasetDict.map` function tailored
    to applying a `DistributedDataPipe` instance to a dataset
    using ray`.

    Arguments:
        self (datasets.Dataset): dataset
        pipe (DistributedDataPipe): data pipe
        unordered (bool):
            whether the order of the dataset should be preserved
        batch_size (int):
            number of examples per batch provided to pipe.
            Defaults to 1000.
        drop_last_batch (bool):
            Whether a last batch smaller than the batch_size
            should be dropped instead of being processed

    Returns:
        transformed_dataset (datasets.IterableDataset):
            dataset after being passed through the data pipe

    """
    return datasets.IterableDatasetDict(
        {
            k: _map_iterable_dataset(
                self=dataset,
                unordered=unordered,
                pipe=pipe,
                batch_size=batch_size,
                drop_last_batch=drop_last_batch,
            )
            for k, dataset in self.items()
        }
    )
