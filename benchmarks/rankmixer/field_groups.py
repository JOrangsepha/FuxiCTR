#!/usr/bin/env python3
"""Ali-CCP field partitions for the T=6 rigor controls.

The 18 categorical fields are the AITM / PaddleRec list, in that order.
``g6_sequential`` cuts this list into 6 contiguous groups of 3.
``g6_random`` applies a fixed Fisher-Yates shuffle and then the same cut.

The partition seed is 42. It is not a training seed. Training seeds
2025–2029 only change initialization and data order. The groups below are
written into ``configs/rigor/model_config.yaml``; training does not shuffle
again. Python guarantees that ``random.Random.random`` is stable across
versions, so the shuffle is implemented with ``random()`` rather than
``Random.shuffle``.
"""

import random


ALI_CCP_FIELDS = [
    "101", "121", "122", "124", "125", "126", "127", "128", "129",
    "205", "206", "207", "216", "508", "509", "702", "853", "301",
]

# Independent of the training seed. Documented in the rigor config and the tech report.
G6_RANDOM_PARTITION_SEED = 42

G6_NUM_GROUPS = 6


def sequential_field_groups(fields, num_groups):
    """Split ``fields`` into ``num_groups`` contiguous chunks of similar size.

    The first ``len(fields) % num_groups`` chunks receive one extra field.
    Ali-CCP has 18 fields and 6 groups, so every chunk has length 3.

    Args:
        fields (list): Field names in table order.
        num_groups (int): Number of groups. Must be at least 1 and at most
            ``len(fields)``.

    Returns:
        list: One list of field names per group. Every input field appears once.
    """
    fields = [str(name) for name in fields]
    if num_groups < 1 or num_groups > len(fields):
        raise ValueError("num_groups must be between 1 and the number of fields.")
    base, extra = divmod(len(fields), num_groups)
    groups = []
    start = 0
    for index in range(num_groups):
        size = base + (1 if index < extra else 0)
        groups.append(fields[start:start + size])
        start += size
    return groups


def deterministic_shuffle(items, seed):
    """Fisher-Yates shuffle driven only by ``random.Random(seed).random``."""
    rng = random.Random(int(seed))
    shuffled = [str(item) for item in items]
    for index in range(len(shuffled) - 1, 0, -1):
        swap = int(rng.random() * (index + 1))
        shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
    return shuffled


def random_field_groups(fields, num_groups, seed):
    """Shuffle ``fields`` with ``seed``, then cut like ``sequential_field_groups``.

    The same seed always returns the same groups. A different seed is a
    different partition. Group sizes match the sequential cut.
    """
    return sequential_field_groups(deterministic_shuffle(fields, seed), num_groups)


def g6_sequential_groups():
    """Six contiguous chunks of the 18 Ali-CCP fields, in feature-list order."""
    return sequential_field_groups(ALI_CCP_FIELDS, G6_NUM_GROUPS)


def g6_random_groups(seed=G6_RANDOM_PARTITION_SEED):
    """Six equal groups from the fixed partition seed (default 42)."""
    return random_field_groups(ALI_CCP_FIELDS, G6_NUM_GROUPS, seed)
