"""Unit tests for evict_policy.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
    TwoQStrategy,
)


def _make_node(**kwargs):
    node = MagicMock()
    node.last_access_time = kwargs.get("last_access_time", 0.0)
    node.hit_count = kwargs.get("hit_count", 0)
    node.creation_time = kwargs.get("creation_time", 0.0)
    node.priority = kwargs.get("priority", 0)
    node.id = kwargs.get("id", 0)
    node.key = kwargs.get("key", [0])
    node.two_q_key = kwargs.get("two_q_key", None)
    node.two_q_device_protected = kwargs.get("two_q_device_protected", False)
    node.two_q_device_seen = kwargs.get("two_q_device_seen", False)
    node.two_q_host_protected = kwargs.get("two_q_host_protected", False)
    node.two_q_host_seen = kwargs.get("two_q_host_seen", False)
    return node


class TestLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LRUStrategy()

    def test_priority_is_last_access_time(self):
        node = _make_node(last_access_time=42.0)
        self.assertEqual(self.strategy.get_priority(node), 42.0)

    def test_older_access_evicted_first(self):
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestLFUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = LFUStrategy()

    def test_priority_is_hit_count_and_time(self):
        node = _make_node(hit_count=5, last_access_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), (5, 3.0))

    def test_lower_hit_count_evicted_first(self):
        cold = _make_node(hit_count=1, last_access_time=10.0)
        hot = _make_node(hit_count=100, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(cold), self.strategy.get_priority(hot)
        )

    def test_same_hit_count_older_access_evicted_first(self):
        old = _make_node(hit_count=3, last_access_time=1.0)
        new = _make_node(hit_count=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestFIFOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FIFOStrategy()

    def test_priority_is_creation_time(self):
        node = _make_node(creation_time=7.0)
        self.assertEqual(self.strategy.get_priority(node), 7.0)

    def test_earlier_created_evicted_first(self):
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(first), self.strategy.get_priority(second)
        )


class TestMRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MRUStrategy()

    def test_priority_is_negated_access_time(self):
        node = _make_node(last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), -5.0)

    def test_most_recently_used_evicted_first(self):
        """MRU evicts the most recently accessed node first (lowest priority value)."""
        old = _make_node(last_access_time=1.0)
        new = _make_node(last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(new), self.strategy.get_priority(old)
        )


class TestFILOStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = FILOStrategy()

    def test_priority_is_negated_creation_time(self):
        node = _make_node(creation_time=3.0)
        self.assertEqual(self.strategy.get_priority(node), -3.0)

    def test_last_created_evicted_first(self):
        """FILO evicts the most recently created node first."""
        first = _make_node(creation_time=1.0)
        second = _make_node(creation_time=5.0)
        self.assertLess(
            self.strategy.get_priority(second), self.strategy.get_priority(first)
        )


class TestPriorityStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = PriorityStrategy()

    def test_priority_is_tuple(self):
        node = _make_node(priority=2, last_access_time=4.0)
        self.assertEqual(self.strategy.get_priority(node), (2, 4.0))

    def test_lower_priority_evicted_first(self):
        low = _make_node(priority=1, last_access_time=10.0)
        high = _make_node(priority=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(low), self.strategy.get_priority(high)
        )

    def test_same_priority_older_access_evicted_first(self):
        old = _make_node(priority=3, last_access_time=1.0)
        new = _make_node(priority=3, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )


class TestSLRUStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = SLRUStrategy(protected_threshold=2)

    def test_probationary_segment(self):
        node = _make_node(hit_count=1, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (0, 5.0))

    def test_protected_segment(self):
        node = _make_node(hit_count=2, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_highly_accessed_is_protected(self):
        node = _make_node(hit_count=100, last_access_time=5.0)
        self.assertEqual(self.strategy.get_priority(node), (1, 5.0))

    def test_probationary_evicted_before_protected(self):
        prob = _make_node(hit_count=1, last_access_time=10.0)
        prot = _make_node(hit_count=5, last_access_time=1.0)
        self.assertLess(
            self.strategy.get_priority(prob), self.strategy.get_priority(prot)
        )

    def test_same_segment_older_access_evicted_first(self):
        old = _make_node(hit_count=0, last_access_time=1.0)
        new = _make_node(hit_count=0, last_access_time=10.0)
        self.assertLess(
            self.strategy.get_priority(old), self.strategy.get_priority(new)
        )

    def test_custom_threshold(self):
        strategy = SLRUStrategy(protected_threshold=5)
        below = _make_node(hit_count=4, last_access_time=1.0)
        at = _make_node(hit_count=5, last_access_time=1.0)
        self.assertEqual(strategy.get_priority(below), (0, 1.0))
        self.assertEqual(strategy.get_priority(at), (1, 1.0))

    def test_default_threshold_is_2(self):
        default = SLRUStrategy()
        self.assertEqual(default.protected_threshold, 2)


class TestTwoQStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TwoQStrategy(a1in_ratio=0.25, ghost_capacity_ratio=1.0)

    def test_second_resident_access_promotes(self):
        node = _make_node(id=1, key=[1] * 4)
        self.strategy.admit(node, "prefix")
        self.strategy.record_access(node)
        self.assertFalse(self.strategy.is_protected(node))
        self.strategy.record_access(node)
        self.assertTrue(self.strategy.is_protected(node))

    def test_probationary_ghost_hit_promotes_on_admission(self):
        node = _make_node(id=1, key=[1] * 4)
        self.strategy.admit(node, "prefix")
        self.strategy.resize_ghost(16)
        self.strategy.record_eviction(node)

        replacement = _make_node(id=2, key=[1] * 4)
        self.strategy.admit(replacement, "prefix")
        self.assertTrue(self.strategy.is_protected(replacement))
        self.assertEqual(self.strategy.ghost_hits["device"], 1)

    def test_a1in_budget_selects_fifo_probation_then_lru_protected(self):
        old_probation = _make_node(
            id=1, key=[1] * 20, creation_time=1, last_access_time=100
        )
        new_probation = _make_node(
            id=2, key=[1] * 20, creation_time=2, last_access_time=101
        )
        old_protected = _make_node(
            id=3,
            key=[1] * 60,
            creation_time=3,
            last_access_time=3,
            two_q_device_protected=True,
        )
        queue = self.strategy.create_eviction_queue(
            [new_probation, old_protected, old_probation]
        )

        def live(_):
            return True

        self.assertIs(queue.pop(live), old_probation)
        # A1in is now 20/80 tokens, exactly its 25% budget, so eviction
        # switches to the LRU protected queue.
        self.assertIs(queue.pop(live), old_protected)

    def test_ghost_queue_is_token_bounded(self):
        self.strategy.resize_ghost(10)
        first = _make_node(id=1, key=[1] * 6)
        second = _make_node(id=2, key=[1] * 6)
        self.strategy.admit(first, "first")
        self.strategy.admit(second, "second")
        self.strategy.record_eviction(first)
        self.strategy.record_eviction(second)
        self.assertNotIn("first", self.strategy._ghost["device"])
        self.assertIn("second", self.strategy._ghost["device"])
        self.assertLessEqual(
            self.strategy._ghost_tokens["device"],
            self.strategy._ghost_capacity_tokens["device"],
        )

    def test_device_eviction_does_not_reset_host_state(self):
        node = _make_node(id=1, key=[1] * 4)
        self.strategy.admit(node, "prefix")
        self.strategy.record_access(node)
        self.strategy.record_access(node)
        self.assertTrue(self.strategy.is_protected(node, tier="device"))
        self.assertTrue(self.strategy.is_protected(node, tier="host"))

        self.strategy.record_eviction(node, tier="device")

        self.assertFalse(self.strategy.is_protected(node, tier="device"))
        self.assertTrue(self.strategy.is_protected(node, tier="host"))

    def test_device_ghost_hit_does_not_change_host_state(self):
        node = _make_node(id=1, key=[1] * 4)
        self.strategy.admit(node, "prefix")
        self.strategy.resize_ghost(16, tier="device")
        self.strategy.record_eviction(node, tier="device")

        self.strategy.record_access(node, tiers=("device",))

        self.assertTrue(self.strategy.is_protected(node, tier="device"))
        self.assertFalse(self.strategy.is_protected(node, tier="host"))
        self.assertEqual(self.strategy.ghost_hits["device"], 1)


class TestEvictionOrdering(unittest.TestCase):
    """Integration-style test: sort a list of nodes by eviction priority."""

    def test_lru_ordering(self):
        strategy = LRUStrategy()
        nodes = [
            _make_node(last_access_time=5.0),
            _make_node(last_access_time=1.0),
            _make_node(last_access_time=3.0),
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        times = [n.last_access_time for n in eviction_order]
        self.assertEqual(times, [1.0, 3.0, 5.0])

    def test_slru_ordering(self):
        strategy = SLRUStrategy(protected_threshold=2)
        nodes = [
            _make_node(hit_count=5, last_access_time=1.0),  # protected, old
            _make_node(hit_count=0, last_access_time=10.0),  # probationary, new
            _make_node(hit_count=0, last_access_time=2.0),  # probationary, old
            _make_node(hit_count=3, last_access_time=8.0),  # protected, new
        ]
        eviction_order = sorted(nodes, key=strategy.get_priority)
        expected = [
            (0, 2.0),  # probationary old
            (0, 10.0),  # probationary new
            (1, 1.0),  # protected old
            (1, 8.0),  # protected new
        ]
        actual = [strategy.get_priority(n) for n in eviction_order]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
