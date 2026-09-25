import os
from pathlib import Path


GIB = 1024**3
DEFAULT_MEMORY_PER_WORKER_BYTES = 4 * GIB
MEMORY_UTILIZATION_NUMERATOR = 4
MEMORY_UTILIZATION_DENOMINATOR = 5


def _physical_memory_bytes() -> int | None:
    sysconf_names = getattr(os, "sysconf_names", {})
    page_size_name = (
        "SC_PAGE_SIZE" if "SC_PAGE_SIZE" in sysconf_names else "SC_PAGESIZE"
    )
    if (
        page_size_name not in sysconf_names
        or "SC_PHYS_PAGES" not in sysconf_names
    ):
        return None
    try:
        memory_bytes = os.sysconf(page_size_name) * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return memory_bytes if memory_bytes > 0 else None


def _cgroup_memory_limit_bytes() -> int | None:
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value == "max":
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        # Some cgroup v1 hosts report an enormous sentinel for "unlimited".
        if 0 < limit < 1 << 60:
            return limit
    return None


def total_memory_bytes() -> int | None:
    """Return usable machine/container RAM when the OS exposes it."""
    limits = [
        limit
        for limit in (_physical_memory_bytes(), _cgroup_memory_limit_bytes())
        if limit is not None
    ]
    return min(limits) if limits else None


def default_worker_count(
    memory_per_worker_bytes: int = DEFAULT_MEMORY_PER_WORKER_BYTES,
) -> int:
    """Choose a conservative process count from CPU capacity and total RAM."""
    if memory_per_worker_bytes <= 0:
        raise ValueError("memory_per_worker_bytes must be positive")

    cpu_workers = max(1, (os.cpu_count() or 1) // 2)
    memory_bytes = total_memory_bytes()
    if memory_bytes is None:
        return cpu_workers

    worker_memory_budget = (
        memory_bytes
        * MEMORY_UTILIZATION_NUMERATOR
        // MEMORY_UTILIZATION_DENOMINATOR
    )
    memory_workers = max(1, worker_memory_budget // memory_per_worker_bytes)
    return min(cpu_workers, memory_workers)
