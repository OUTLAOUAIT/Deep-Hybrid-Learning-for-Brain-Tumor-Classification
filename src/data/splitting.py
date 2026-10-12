"""Leakage-conscious split creation and integrity checks."""
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path


def _identities(row, root):
    path = Path(root) / row["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identities = ["sha256:" + digest]
    if row.get("group"):
        identities.append("group:" + str(row["group"]))
    return identities


def create_splits(rows, root, output_dir, seed=42, ratios=(0.8, 0.1, 0.1)):
    """Group exact duplicates and declared subjects; approximate class balance by groups."""
    if abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) <= 0:
        raise ValueError("Split ratios must be positive and sum to one")
    # Union samples sharing a declared group OR exact file digest. This also catches
    # identical files whose metadata happens to declare different groups.
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    owner = {}
    for index, row in enumerate(rows):
        for identity in _identities(row, root):
            if identity in owner:
                union(index, owner[identity])
            else:
                owner[identity] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    by_class = defaultdict(list)
    for key, members in groups.items():
        labels = {str(row["label"]) for row in members}
        if len(labels) != 1:
            raise ValueError("A duplicate/group identity has conflicting labels")
        by_class[next(iter(labels))].append((key, members))
    rng = random.Random(seed)
    assigned = [[], [], []]
    for label in sorted(by_class):
        units = by_class[label]
        rng.shuffle(units)
        n = len(units)
        if n < 3:
            raise ValueError("Each class needs at least three independent groups")
        counts = [max(1, int(n * ratios[0])), max(1, int(n * ratios[1]))]
        if sum(counts) >= n:
            counts = [n - 2, 1]
        boundaries = (counts[0], counts[0] + counts[1])
        for i, (_, members) in enumerate(units):
            bucket = 0 if i < boundaries[0] else (1 if i < boundaries[1] else 2)
            assigned[bucket].extend(members)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"), assigned):
        with open(out / (name + ".csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(part)
    validate_splits(assigned, root)
    return assigned


def validate_splits(splits, root):
    seen = set()
    for split in splits:
        for row in split:
            identities = set(_identities(row, root))
            if identities & seen:
                raise ValueError("Group or exact duplicate crosses split boundaries")
            seen.update(identities)
    return True
