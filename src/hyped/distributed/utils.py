"""Helper functionalities."""
from importlib import import_module

from ray.actor import ActorHandle


def get_actor_type(actor: ActorHandle) -> type:
    """Get the underlying type of a ray remote actor.

    Arguments:
        actor (ActorHandle): actor

    Returns:
        t (type): type of the actor
    """
    # get class and module name from actor handle
    class_name = actor._ray_actor_creation_function_descriptor.class_name
    module_name = actor._ray_actor_creation_function_descriptor.module_name
    # import module and get type
    module = import_module(module_name)
    return getattr(module, class_name)
