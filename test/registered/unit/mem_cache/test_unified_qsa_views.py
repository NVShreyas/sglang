"""Addressing and page-lifecycle tests for QSA in the unified device pool."""

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_mha_views,
    build_page_major_qsa_mha_views,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    DenseDraftRegion,
    QSAMHASubPoolSpec,
    UnifiedQSAMHATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestUnifiedQSAViews(unittest.TestCase):
    PAGE = 64
    RATIO = 4

    def setUp(self):
        self.spec = QSAMHASubPoolSpec(
            name="full",
            layer_num=3,
            head_num=2,
            head_dim=8,
            store_dtype=torch.bfloat16,
            grow_direction="down",
            qsa_index_kv_heads=1,
            qsa_index_head_dim=16,
            qsa_compress_ratio=self.RATIO,
        )
        self.num_pages = 5
        self.raw = torch.zeros(
            self.num_pages * self.spec.page_bytes(self.PAGE), dtype=torch.uint8
        )
        self.k, self.v, self.qsa = build_page_major_qsa_mha_views(
            self.raw,
            layer_num=self.spec.layer_num,
            head_num=self.spec.head_num,
            head_dim=self.spec.head_dim,
            v_head_dim=self.spec.v_head_dim,
            store_dtype=self.spec.store_dtype,
            qsa_index_kv_heads=self.spec.qsa_index_kv_heads,
            qsa_index_head_dim=self.spec.qsa_index_head_dim,
            qsa_store_dtype=self.spec.qsa_store_dtype,
            qsa_compress_ratio=self.spec.qsa_compress_ratio,
            page_size=self.PAGE,
            num_pages=self.num_pages,
        )

    def _qsa_rows(self, layer, compressed_locs):
        cps = self.PAGE // self.RATIO
        locs = torch.as_tensor(compressed_locs)
        pages, within = locs // cps, locs % cps
        return self.spec.qsa_page_index(pages, layer, self.PAGE) * cps + within

    def test_byte_accounting_includes_compressed_keys(self):
        mha = self.spec.layer_num * (self.spec.k_row_bytes() + self.spec.v_row_bytes())
        qsa = self.spec.layer_num * self.spec.qsa_row_bytes() // self.RATIO
        self.assertEqual(self.spec.entry_bytes(), mha + qsa)

    def test_qsa_translation_uses_the_inherited_page_size(self):
        pool = UnifiedQSAMHATokenToKVPool.__new__(UnifiedQSAMHATokenToKVPool)
        pool.page_size = self.PAGE
        pool._qsa_spec = self.spec
        loc = torch.tensor([0, 15, 16, 31], dtype=torch.int64)

        translated = pool.translate_qsa_compressed_locs(1, loc)

        cps = self.PAGE // self.RATIO
        expected = self.spec.qsa_page_index(loc // cps, 1, self.PAGE) * cps + loc % cps
        torch.testing.assert_close(translated, expected, rtol=0, atol=0)

    def test_qsa_store_converts_dense_ids_to_page_offsets(self):
        pool = UnifiedQSAMHATokenToKVPool.__new__(UnifiedQSAMHATokenToKVPool)
        pool.page_size = self.PAGE
        pool._qsa_spec = self.spec
        pool.k_buffer = self.k
        pool.v_buffer = self.v
        page, offset = 2, 7
        dense_loc = torch.tensor(
            [page * self.PAGE * self.spec.blocks_per_page() + offset]
        )

        pool._store_kv_layer(
            1,
            dense_loc,
            torch.full((1, 2, 8), 3.0, dtype=torch.bfloat16),
            torch.full((1, 2, 8), 5.0, dtype=torch.bfloat16),
        )

        self.assertTrue(torch.all(self.k[1][page, offset] == 3))
        self.assertTrue(torch.all(self.v[1][page, offset] == 5))

    def test_kv_and_qsa_regions_do_not_alias(self):
        for layer in range(self.spec.layer_num):
            self.k[layer].zero_()
            self.v[layer].zero_()
        for layer in range(self.spec.layer_num):
            rows = self._qsa_rows(layer, torch.arange(self.num_pages * 16))
            self.qsa[rows] = layer + 1
        for layer in range(self.spec.layer_num):
            self.assertTrue(torch.count_nonzero(self.k[layer]) == 0)
            self.assertTrue(torch.count_nonzero(self.v[layer]) == 0)
            rows = self._qsa_rows(layer, torch.arange(self.num_pages * 16))
            self.assertTrue(torch.all(self.qsa[rows] == layer + 1))

    def test_page_envelope_move_carries_kv_and_qsa(self):
        src, dst = 4, 1
        self.k[0][src].fill_(3)
        self.v[2][src].fill_(5)
        qsa_rows = self._qsa_rows(1, torch.arange(src * 16, (src + 1) * 16))
        self.qsa[qsa_rows] = 7
        pages = self.raw.view(self.num_pages, self.spec.page_bytes(self.PAGE))
        pages[dst] = pages[src].clone()
        self.assertTrue(torch.all(self.k[0][dst] == 3))
        self.assertTrue(torch.all(self.v[2][dst] == 5))
        dst_rows = self._qsa_rows(1, torch.arange(dst * 16, (dst + 1) * 16))
        self.assertTrue(torch.all(self.qsa[dst_rows] == 7))

    def test_fused_draft_region_composes_with_qsa_envelope(self):
        draft = DenseDraftRegion(
            layer_num=2,
            head_num=1,
            head_dim=8,
            store_dtype=torch.bfloat16,
        )
        spec = QSAMHASubPoolSpec(
            name="full",
            layer_num=3,
            head_num=2,
            head_dim=8,
            store_dtype=torch.bfloat16,
            grow_direction="down",
            qsa_index_kv_heads=1,
            qsa_index_head_dim=16,
            qsa_compress_ratio=self.RATIO,
            draft_region=draft,
        )
        num_pages = 3
        page_bytes = spec.page_bytes(self.PAGE)
        raw = torch.zeros(
            num_pages * page_bytes + spec.view_tail_pad_bytes(self.PAGE),
            dtype=torch.uint8,
        )
        k, v, qsa = build_page_major_qsa_mha_views(
            raw,
            layer_num=spec.layer_num,
            head_num=spec.head_num,
            head_dim=spec.head_dim,
            v_head_dim=spec.v_head_dim,
            store_dtype=spec.store_dtype,
            qsa_index_kv_heads=spec.qsa_index_kv_heads,
            qsa_index_head_dim=spec.qsa_index_head_dim,
            qsa_store_dtype=spec.qsa_store_dtype,
            qsa_compress_ratio=spec.qsa_compress_ratio,
            page_size=self.PAGE,
            num_pages=num_pages,
            page_stride_bytes=page_bytes,
        )
        dk, dv = build_mha_views(
            raw,
            layer_num=draft.layer_num,
            head_num=draft.head_num,
            head_dim=draft.head_dim,
            v_head_dim=draft.head_dim,
            store_dtype=draft.store_dtype,
            page_size=self.PAGE,
            num_pages=num_pages,
            page_stride_blocks=spec.draft_kernel_page_multiplier(),
            region_offset_bytes=spec.draft_region_offset_in_page(self.PAGE),
        )

        page = 1
        cps = self.PAGE // self.RATIO
        qsa_rows = spec.qsa_page_index(
            torch.tensor([page]), 1, self.PAGE
        ) * cps + torch.arange(cps)
        qsa[qsa_rows] = 7
        draft_id = page * self.PAGE * spec.draft_kernel_page_multiplier()
        dk[0][draft_id].fill_(3)
        dv[1][draft_id].fill_(5)

        self.assertTrue(torch.all(qsa[qsa_rows] == 7))
        self.assertTrue(torch.all(dk[0][draft_id] == 3))
        self.assertTrue(torch.all(dv[1][draft_id] == 5))
        self.assertTrue(torch.count_nonzero(k[0][page]) == 0)
        self.assertTrue(torch.count_nonzero(v[2][page]) == 0)


if __name__ == "__main__":
    unittest.main()
