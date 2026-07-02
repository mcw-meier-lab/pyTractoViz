from __future__ import annotations

from typing import Any

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]


def _get_resource_usage() -> Any | None:
    if resource is None:
        return None

    getrusage = getattr(resource, "getrusage", None)
    rusage_self = getattr(resource, "RUSAGE_SELF", None)
    if getrusage is None or rusage_self is None:
        return None

    return getrusage(rusage_self)


def _set_resource_memory_limit(memory_limit_bytes: int) -> bool:
    if resource is None:
        return False

    setrlimit = getattr(resource, "setrlimit", None)
    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    rlim_infinity = getattr(resource, "RLIM_INFINITY", None)
    if setrlimit is None or rlimit_as is None or rlim_infinity is None:
        return False

    setrlimit(rlimit_as, (memory_limit_bytes, rlim_infinity))
    return True
