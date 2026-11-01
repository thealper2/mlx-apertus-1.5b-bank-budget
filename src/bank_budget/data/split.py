"""Which lines go where (part of the experiment definition).

1. **Copies.** Lines whose text carries two different categories are dropped; every other
   text is kept once (its first occurrence).
2. **Shops.** Every line gets a shop group (:func:`bank_budget.task.shop_groups`).
3. **Held-out shops.** With ``random.Random(holdout_seed)``, per category (in the order of
   ``task.CATEGORIES``): the groups whose most frequent category it is, in sorted order,
   are shuffled and taken one by one until they hold ``holdout_share`` of those groups'
   lines; a group with more than ``holdout_max_group_share`` of them is skipped.
4. **Tests.** ``test_new``: ``test_per_category`` lines per category from held-out shops
   (seed ``test_new_seed``); ``test_known``: as many from the other shops
   (``test_known_seed``) — new lines of shops the model trains on.
5. **Training.** The remaining lines of the other shops: ``valid_per_category`` per category
   for ``valid`` (``valid_seed``), the rest is ``train``, shuffled with
   ``train_shuffle_seed``.

"Per category" sampling draws ``min(n, available)`` lines of each category (categories in
sorted order) with ``DataFrame.sample(random_state=seed)``.
"""

from __future__ import annotations

import random
from typing import Any

from bank_budget.config import SplitConfig
from bank_budget.task import CATEGORIES, shop_groups


def deduplicate(frame: Any) -> Any:
    """One row per text; texts filed under two categories are dropped entirely."""
    per_text = frame.groupby("description").category.nunique()
    ambiguous = per_text[per_text > 1].index
    kept = frame[~frame.description.isin(ambiguous)].drop_duplicates("description")
    return kept.reset_index(drop=True)


def held_out_groups(frame: Any, share: float, max_group_share: float, seed: int) -> set[str]:
    """Shop groups kept out of training entirely (step 3 of the module docstring)."""
    rng = random.Random(seed)
    majority = frame.groupby("group").category.agg(lambda s: s.value_counts().index[0])
    size = frame.groupby("group").size()
    held: set[str] = set()
    for category in CATEGORIES:
        groups = list(majority[majority == category].index)
        rng.shuffle(groups)
        total, taken = size[groups].sum(), 0
        for g in groups:
            if taken >= share * total:
                break
            if size[g] > max_group_share * total:
                continue
            held.add(g)
            taken += size[g]
    return held


def per_category(frame: Any, n: int, seed: int) -> Any:
    """Up to ``n`` lines of every category (sorted category order)."""
    import pandas as pd

    parts = [g.sample(min(n, len(g)), random_state=seed) for _, g in frame.groupby("category")]
    return pd.concat(parts)


def make_splits(lines: Any, cfg: SplitConfig) -> dict[str, Any]:
    """The four splits as DataFrames with ``description``, ``category`` and ``group``."""
    frame = deduplicate(lines)
    frame["group"] = shop_groups(list(frame.description))
    held = held_out_groups(frame, cfg.holdout_share, cfg.holdout_max_group_share, cfg.holdout_seed)
    new_shops = frame[frame.group.isin(held)]
    known_shops = frame[~frame.group.isin(held)]
    test_new = per_category(new_shops, cfg.test_per_category, cfg.test_new_seed)
    test_known = per_category(known_shops, cfg.test_per_category, cfg.test_known_seed)
    rest = known_shops.drop(test_known.index)
    valid = per_category(rest, cfg.valid_per_category, cfg.valid_seed)
    train = rest.drop(valid.index).sample(frac=1, random_state=cfg.train_shuffle_seed)
    return {
        "unique": frame,
        "held_out_groups": held,
        "train": train,
        "valid": valid,
        "test_new": test_new,
        "test_known": test_known,
    }


def to_rows(frame: Any) -> list[dict[str, str]]:
    """``{line, category, shop}`` dicts in frame order."""
    return [
        {"line": d, "category": c, "shop": g}
        for d, c, g in zip(frame.description, frame.category, frame.group, strict=True)
    ]
