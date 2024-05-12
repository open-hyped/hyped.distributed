"""Ray Remote Actor Pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

import ray
from ray.actor import ActorHandle
from ray.util.queue import Empty, Queue

from .utils import get_actor_type

T = TypeVar("T")


# global variable used communicate the worker rank
# between objects
rank: None | int = None


@dataclass
class RemoteWorker(object):
    """Remote Worker Class.

    Base class of remote actor workers. Keeps track of the rank of
    the remote worker.

    Attributes:
        rank (int): rank of the actor
    """

    rank: int

    def get_rank(self) -> int:
        """Get the rank of the worker.

        Returns:
            rank (int): rank of the remote actor
        """
        return self.rank

    def ping(self) -> None:
        """Ping.

        Used to wait for this actor to be initialized.
        Also sets the global rank variable.
        """
        global rank
        rank = self.get_rank()


@dataclass
class _ReservedActor(object):
    """Helper Class."""

    _pool: ActorPool
    _rank: int
    _actor: ActorHandle

    _is_released: bool = field(init=False, default=False)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def actor(self) -> ActorHandle:
        return self._actor

    def __iter__(self):
        return iter([self.rank, self.actor])

    def release(self) -> None:
        if not self._is_released:
            self._pool._release(self.rank)
            self._is_released = True

    def __enter__(self) -> _ReservedActor:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()


class _ReservedActorList(list[_ReservedActor]):
    """Helper Class."""

    @property
    def ranks(self) -> list[int]:
        return [a.rank for a in self]

    @property
    def actors(self) -> list[ActorHandle]:
        return [a.actor for a in self]

    def for_all_actors(self, fn: Callable[[ActorHandle], T]) -> list[T]:
        return list(map(fn, self.actors))

    def release(self):
        for actor in self:
            actor.release()

    def __enter__(self) -> _ReservedActorList:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()


class ActorPool(object):
    """Ray Actor Pool.

    Helper class managing a list of ray actors

    Arguments:
        actors (list[ActorHandle]): pool of ray actors
    """

    def __init__(self, actors: list[ActorHandle]) -> None:
        """Constructor."""
        assert len(actors) > 0, "No actors provided"

        # collect all actor types
        actor_types = set(list(map(get_actor_type, actors)))

        if len(actor_types) > 1:
            raise RuntimeError(
                "Expected all actors to be of the same type, got %s"
                % actor_types
            )

        # unpack the actor types
        (actor_type,) = actor_types

        if not issubclass(actor_type, RemoteWorker):
            raise TypeError(
                "All actors used in an actor pool must inherit the "
                "`RemoteWorker` type, got %s" % str(actor_type)
            )

        ranks = ray.get([a.get_rank.remote() for a in actors])
        self.actors = dict(zip(ranks, actors))
        # create idle rank queue used to get ranks waiting for work
        self.idle_ranks = Queue(maxsize=len(actors))
        self.idle_ranks.put_nowait_batch(ranks)

        # ping all actors
        ray.wait(
            [actor.ping.remote() for actor in actors], num_returns=len(actors)
        )

    @property
    def num_actors(self) -> int:
        """Number of actors in the pool."""
        return len(self.actors)

    def reserve(
        self, block: bool = True, timeout: float | None = None
    ) -> _ReservedActor:
        """Reserve a single actor in the pool.

        Arguments:
            block (bool):
                whether to block until an actor is idleing.
                Defaults to True.
            timeout (float):
                a timeout of how long to block before throwing
                a `ray.util.queue.Empty` exception.
            block (bool):
                whether to block until actor is reserved
            timeout (None | float):
                blocks at most timeout seconds and raises the Empty
                exception if no actor could be reserved

        Returns:
            actor (_ReservedActor): reserved actor
        """
        # get idleing actor rank from queue and
        # the corresponsing actor
        rank = self.idle_ranks.get(**kwargs)
        actor = self.actors[rank]
        # return rank and actor
        return _ReservedActor(self, rank, actor)

    def reserve_all(self) -> _ReservedActorList:
        """Reserve all currently idleing actors in the pool.

        Returns:
            actors (_ReservedActorList):
                list of all reserved actors
        """
        reserved_actors = _ReservedActorList()

        try:
            # collect all idleing actors
            while not self.idle_ranks.empty():
                actor = self.reserve(block=False)
                reserved_actors.append(actor)
        except Empty:
            pass

        return reserved_actors

    def _release(self, rank: int) -> None:
        """Release reserved rank.

        Internal function to release a reserved
        actor of a given rank.

        Arguments:
            rank (int): rank of the actor to release
        """
        # put rank back to idle queue
        assert rank < self.num_actors
        self.idle_ranks.put(rank)

    def destroy(self) -> None:
        """Destroy all actors of the pool."""
        for actor in self.actors:
            ray.kill(actor)

    @staticmethod
    def reserve_from_each(
        pools: list[ActorPool], timeout: None | float = None
    ) -> Iterable[_ReservedActor]:
        """Reserve one actor from each pool.

        Arguments:
            pools (list[ActorPool]):
                list of actor pools to reserve actors from
            timeout (None | float):
                blocks at most timeout seconds and raises the Empty exception

        Returns:
            actor_iter (Iterable[_ReservedActor]):
                Iterable over reserved actors
        """
        tasks = [
            pool.idle_ranks.actor.get.remote(timeout=timeout) for pool in pools
        ]
        task2idx = {t: i for i, t in enumerate(tasks)}

        while len(tasks) > 0:
            done, tasks = ray.wait(tasks, num_returns=1)

            for idx, rank in zip(map(task2idx.get, done), ray.get(done)):
                assert idx is not None
                pool = pools[idx]
                yield idx, _ReservedActor(pool, rank, pool.actors[rank])
