"""Shared exception identity across script, module and spawned worker imports."""


class CollectionCutoff(RuntimeError):
    """A resource cutoff, never a completed population or task termination."""
