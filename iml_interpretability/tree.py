"""Library-independent tree representation that interpretability metrics operate on."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.tree import BaseDecisionTree


@dataclass(frozen=True)
class Node:
    feature: int | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def __str__(self):
        return self.render()

    def render(self, depth: int = 0) -> str:
        prefix = "  " * depth
        if self.is_leaf:
            return prefix + "leaf"
        return "\n".join([prefix + f"x{self.feature}", self.left.render(depth + 1), self.right.render(depth + 1)])


def from_sklearn(model: BaseDecisionTree) -> Node:
    if not isinstance(model, BaseDecisionTree):
        raise TypeError(f"Only supports BaseDecisionTree, got {type(model).__name__}")

    tree = model.tree_

    def build(node_id: int) -> Node:
        # sklearn marks leaves with -1
        if tree.children_left[node_id] == -1:
            return Node()
        return Node(
            feature=int(tree.feature[node_id]),
            left=build(tree.children_left[node_id]),
            right=build(tree.children_right[node_id]),
        )

    return build(0)


def n_leaves(node: Node) -> int:
    return 1 if node.is_leaf else n_leaves(node.left) + n_leaves(node.right)


def depth(node: Node) -> int:
    return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))


def max_features_on_path(node: Node, seen: frozenset[int] = frozenset()) -> int:
    """Largest number of distinct features tested on a single root-to-leaf path.

    :param seen: Features already tested above ``node``; leave empty when calling on the root.
    """
    if node.is_leaf:
        return len(seen)
    seen = seen | {node.feature}
    return max(max_features_on_path(node.left, seen), max_features_on_path(node.right, seen))


def feature_reuse(node: Node, seen: frozenset[int] = frozenset()) -> int:
    """Number of splits on a feature that an ancestor of the split already tested.

    :param seen: Features already tested above ``node``; leave empty when calling on the root.
    """
    if node.is_leaf:
        return 0
    reused = int(node.feature in seen)
    seen = seen | {node.feature}
    return reused + feature_reuse(node.left, seen) + feature_reuse(node.right, seen)
