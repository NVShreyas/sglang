from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import heapq
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class TwoQEvictionQueue:
    """Token-weighted A1in/Am victim queues for one radix eviction walk."""

    def __init__(self, strategy: TwoQStrategy, nodes: Iterable[TreeNode], tier: str):
        self.strategy = strategy
        self.tier = tier
        self.probationary = []
        self.protected = []
        self.probationary_tokens = 0
        self.protected_tokens = 0
        for node in nodes:
            self.push(node)
        strategy.resize_ghost(
            self.probationary_tokens + self.protected_tokens, tier=tier
        )

    @staticmethod
    def _weight(node: TreeNode) -> int:
        return len(node.key)

    def push(self, node: TreeNode) -> None:
        weight = self._weight(node)
        if self.strategy.is_protected(node, tier=self.tier):
            heapq.heappush(
                self.protected, (node.last_access_time, node.id, weight, node)
            )
            self.protected_tokens += weight
        else:
            heapq.heappush(
                self.probationary, (node.creation_time, node.id, weight, node)
            )
            self.probationary_tokens += weight

    def _pop_from(
        self,
        *,
        protected: bool,
        is_live: Callable[[TreeNode], bool],
    ) -> Optional[TreeNode]:
        heap = self.protected if protected else self.probationary
        while heap:
            _, _, weight, node = heapq.heappop(heap)
            if protected:
                self.protected_tokens -= weight
            else:
                self.probationary_tokens -= weight
            if not is_live(node):
                continue
            if self.strategy.is_protected(node, tier=self.tier) != protected:
                self.push(node)
                continue
            return node
        return None

    def pop(self, is_live: Callable[[TreeNode], bool]) -> Optional[TreeNode]:
        total = self.probationary_tokens + self.protected_tokens
        evict_probationary = self.probationary_tokens > self.strategy.a1in_ratio * total
        node = self._pop_from(protected=not evict_probationary, is_live=is_live)
        if node is None:
            node = self._pop_from(protected=evict_probationary, is_live=is_live)
        return node


class TwoQStrategy(EvictionStrategy):
    """2Q adapted to variable-sized radix nodes and independent cache tiers."""

    _TIERS = ("device", "host")

    def __init__(self, a1in_ratio: float = 0.25, ghost_capacity_ratio: float = 1.0):
        if not 0 < a1in_ratio < 1:
            raise ValueError(f"a1in_ratio must be between 0 and 1, got {a1in_ratio}")
        if ghost_capacity_ratio <= 0:
            raise ValueError(
                f"ghost_capacity_ratio must be positive, got {ghost_capacity_ratio}"
            )
        self.a1in_ratio = a1in_ratio
        self.ghost_capacity_ratio = ghost_capacity_ratio
        self.reset()

    def reset(self) -> None:
        self._ghost: dict[str, OrderedDict[Hashable, int]] = {
            tier: OrderedDict() for tier in self._TIERS
        }
        self._ghost_tokens = {tier: 0 for tier in self._TIERS}
        self._ghost_capacity_tokens = {tier: 0 for tier in self._TIERS}
        self.ghost_hits = {tier: 0 for tier in self._TIERS}
        self.promotions = 0

    @staticmethod
    def is_protected(node: TreeNode, tier: str = "device") -> bool:
        return bool(getattr(node, f"two_q_{tier}_protected", False))

    def admit(self, node: TreeNode, key: Hashable) -> None:
        node.two_q_key = key
        for tier in self._TIERS:
            ghost = self._ghost[tier]
            weight = ghost.pop(key, None)
            if weight is not None:
                self._ghost_tokens[tier] -= weight
                setattr(node, f"two_q_{tier}_protected", True)
                setattr(node, f"two_q_{tier}_seen", True)
                self.ghost_hits[tier] += 1
                self.promotions += 1
            else:
                setattr(node, f"two_q_{tier}_protected", False)
                setattr(node, f"two_q_{tier}_seen", False)

    def inherit(self, new_node: TreeNode, child: TreeNode, key: Hashable) -> None:
        new_node.two_q_key = key
        for tier in self._TIERS:
            setattr(
                new_node,
                f"two_q_{tier}_protected",
                self.is_protected(child, tier=tier),
            )
            setattr(
                new_node,
                f"two_q_{tier}_seen",
                bool(getattr(child, f"two_q_{tier}_seen", False)),
            )

    def _record_tier_access(self, node: TreeNode, tier: str) -> None:
        if self.is_protected(node, tier=tier):
            return
        key = getattr(node, "two_q_key", None)
        ghost = self._ghost[tier]
        ghost_weight = ghost.pop(key, None) if key is not None else None
        if ghost_weight is not None:
            self._ghost_tokens[tier] -= ghost_weight
            self.ghost_hits[tier] += 1
            setattr(node, f"two_q_{tier}_protected", True)
            setattr(node, f"two_q_{tier}_seen", True)
            self.promotions += 1
        elif getattr(node, f"two_q_{tier}_seen", False):
            setattr(node, f"two_q_{tier}_protected", True)
            self.promotions += 1
        else:
            setattr(node, f"two_q_{tier}_seen", True)

    def record_access(self, node: TreeNode, tiers: Iterable[str] = _TIERS) -> None:
        for tier in tiers:
            self._record_tier_access(node, tier)

    def record_eviction(self, node: TreeNode, tier: str = "device") -> None:
        key = getattr(node, "two_q_key", None)
        ghost = self._ghost[tier]
        if key is not None and not self.is_protected(node, tier=tier):
            old_weight = ghost.pop(key, None)
            if old_weight is not None:
                self._ghost_tokens[tier] -= old_weight
            weight = len(node.key)
            ghost[key] = weight
            self._ghost_tokens[tier] += weight
            self._trim_ghost(tier)
        setattr(node, f"two_q_{tier}_protected", False)
        setattr(node, f"two_q_{tier}_seen", False)

    def resize_ghost(self, resident_tokens: int, tier: str = "device") -> None:
        self._ghost_capacity_tokens[tier] = max(
            1, int(resident_tokens * self.ghost_capacity_ratio)
        )
        self._trim_ghost(tier)

    def _trim_ghost(self, tier: str) -> None:
        ghost = self._ghost[tier]
        while self._ghost_tokens[tier] > self._ghost_capacity_tokens[tier] and ghost:
            _, weight = ghost.popitem(last=False)
            self._ghost_tokens[tier] -= weight

    def create_eviction_queue(
        self, nodes: Iterable[TreeNode], tier: str = "device"
    ) -> TwoQEvictionQueue:
        return TwoQEvictionQueue(self, nodes, tier)

    def get_priority(self, node: TreeNode) -> tuple[int, float]:
        if self.is_protected(node):
            return (1, node.last_access_time)
        return (0, node.creation_time)
