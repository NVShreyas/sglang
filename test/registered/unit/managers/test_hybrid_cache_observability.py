import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.utils import cached_tokens_details_from_dict
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    get_hybrid_cache_details,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.observability.metrics_collector import SchedulerStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHybridCacheObservability(unittest.TestCase):
    def test_reports_per_pool_host_occupancy_without_tree_scan(self):
        def pool(total, available):
            return SimpleNamespace(logical_size=total, available_size=lambda: available)

        kv_pool = pool(100, 25)
        mamba_pool = pool(20, 12)
        reporter = SchedulerMetricsReporter.__new__(SchedulerMetricsReporter)
        reporter.stats = SchedulerStats()
        reporter.scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            tree_cache=SimpleNamespace(
                token_to_kv_pool_host=kv_pool,
                host_pool_group=SimpleNamespace(
                    entries=[
                        SimpleNamespace(name=PoolName.KV, host_pool=kv_pool),
                        SimpleNamespace(name=PoolName.MAMBA, host_pool=mamba_pool),
                    ]
                ),
            ),
        )

        reporter._log_hicache_stats()

        self.assertEqual(
            reporter.stats.hicache_host_pool_used_slots, {"kv": 75, "mamba": 8}
        )
        self.assertEqual(
            reporter.stats.hicache_host_pool_total_slots, {"kv": 100, "mamba": 20}
        )

    def test_aggregates_host_evictions_once_per_component(self):
        calls = []
        result = SimpleNamespace(
            tracker={ComponentType.FULL: 64, ComponentType.MAMBA: 2},
            device_frees={},
            host_frees={
                ComponentType.FULL: [object()],
                ComponentType.MAMBA: [object(), object()],
            },
        )
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.host_memory_mode = "cache"
        cache.tree_core = SimpleNamespace(
            drive_host_eviction=lambda component_type, num_tokens: result
        )
        cache.metrics_collector = SimpleNamespace(
            increment_hicache_host_evictions=lambda **kwargs: calls.append(kwargs)
        )
        cache._free_values = lambda device_frees, host_frees: (
            device_frees.clear(),
            host_frees.clear(),
        )

        freed = cache.evict_host(64, ComponentType.FULL)

        self.assertEqual(freed, 64)
        self.assertCountEqual(
            calls,
            [
                {"pool": "kv", "trigger_pool": "kv", "slots": 64, "nodes": 1},
                {
                    "pool": "mamba",
                    "trigger_pool": "kv",
                    "slots": 2,
                    "nodes": 2,
                },
            ],
        )

    def test_classifies_checkpoint_source_before_kv_load_back(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.components = {ComponentType.MAMBA: object()}
        device_values = {1: object(), 2: None, 3: None}
        cache.tree_core = SimpleNamespace(
            get_component_device_value=lambda node_id, component_type: device_values[
                node_id
            ]
        )

        self.assertEqual(cache._classify_load_back_checkpoint(1, {}), "device_resident")
        self.assertEqual(
            cache._classify_load_back_checkpoint(
                2,
                {ComponentType.MAMBA: [PoolTransfer(name=PoolName.MAMBA)]},
            ),
            "host_restored",
        )
        self.assertEqual(cache._classify_load_back_checkpoint(3, {}), "unavailable")

        cache.components = {}
        self.assertEqual(cache._classify_load_back_checkpoint(3, {}), "not_applicable")

    def test_reports_raw_joint_trimmed_and_compute_tokens(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(21)),
            cached_tokens=8,
            cached_tokens_device=8,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            hybrid_cache_full_attention_cached_tokens=16,
            hybrid_cache_recurrent_state_cached_tokens=8,
        )

        details = get_hybrid_cache_details(req)

        self.assertEqual(details["full_attention_cached_tokens"], 16)
        self.assertEqual(details["recurrent_state_cached_tokens"], 8)
        self.assertEqual(details["usable_cached_tokens"], 8)
        self.assertEqual(details["trimmed_full_attention_tokens"], 8)
        self.assertEqual(details["full_attention_recomputed_tokens"], 13)
        self.assertEqual(details["recurrent_recomputed_tokens"], 13)
        self.assertEqual(details["recurrent_replayed_tokens"], 0)

    def test_native_and_openai_details_preserve_hybrid_fields(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(9)),
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            hybrid_cache_full_attention_cached_tokens=0,
            hybrid_cache_recurrent_state_cached_tokens=0,
        )
        streamer = SimpleNamespace(enable_hicache_storage=lambda: False)

        details = SchedulerOutputStreamer.get_cached_tokens_details(streamer, req)
        wire_details = cached_tokens_details_from_dict(details).model_dump(
            exclude_none=True
        )

        self.assertEqual(wire_details, details)
        self.assertEqual(wire_details["usable_cached_tokens"], 0)
        self.assertEqual(wire_details["full_attention_recomputed_tokens"], 9)

    def test_non_hybrid_request_keeps_existing_shape(self):
        req = SimpleNamespace(
            origin_input_ids=list(range(9)),
            cached_tokens=4,
            cached_tokens_device=4,
            cached_tokens_host=0,
            cached_tokens_storage=0,
        )
        streamer = SimpleNamespace(enable_hicache_storage=lambda: False)

        details = SchedulerOutputStreamer.get_cached_tokens_details(streamer, req)

        self.assertEqual(details, {"device": 4, "host": 0})


if __name__ == "__main__":
    unittest.main()
