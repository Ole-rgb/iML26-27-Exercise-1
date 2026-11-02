"""Tests for the structural tree features in :mod:`iml_interpretability.tree`.

Every feature is checked on the same small hand-built trees, so each expected value can be counted by hand.
The feature functions are looked up as ``tree.<name>`` at test time: until a function exists in ``tree.py``,
its test fails with an ``AttributeError`` naming the missing function.
"""

from __future__ import annotations

import pytest

from iml_interpretability import tree
from iml_interpretability.tree import Node

LEAF = Node()

STUMP = Node(0, LEAF, LEAF)

# imbalanced tree
CHAIN = Node(0, LEAF, Node(1, LEAF, Node(2, LEAF, LEAF)))

# complete tree of depth 3
COMPLETE = Node(0, Node(1, Node(2, LEAF, LEAF), Node(2, LEAF, LEAF)), Node(1, Node(2, LEAF, LEAF), Node(2, LEAF, LEAF)))

# reuses x0
REUSE = Node(0, Node(0, LEAF, LEAF), Node(1, LEAF, Node(0, LEAF, LEAF)))

TREES = {"leaf": LEAF, "stump": STUMP, "chain": CHAIN, "complete": COMPLETE, "reuse": REUSE}


def _cases(expected: dict[str, int]) -> list:
    """Turn ``{tree name: expected value}`` into parametrize cases with readable ids."""
    return [pytest.param(TREES[name], value, id=name) for name, value in expected.items()]


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 1, "stump": 2, "chain": 4, "complete": 8, "reuse": 5}))
def test_n_leaves(node, expected):
    assert tree.n_leaves(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 1, "chain": 3, "complete": 3, "reuse": 3}))
def test_depth(node, expected):
    assert tree.depth(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 1, "chain": 3, "complete": 3, "reuse": 2}))
def test_max_features_on_path(node, expected):
    """Largest number of *distinct* features on a single root-to-leaf path."""
    assert tree.max_features_on_path(node) == expected


@pytest.mark.parametrize(("node", "expected"), _cases({"leaf": 0, "stump": 0, "chain": 0, "complete": 0, "reuse": 2}))
def test_feature_reuse(node, expected):
    """Number of splits on a feature that an ancestor already tested."""
    assert tree.feature_reuse(node) == expected
