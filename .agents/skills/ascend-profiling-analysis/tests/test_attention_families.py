"""Family-resolution tests for attention.

We don't have a YAML loader yet, so the cheat-sheet from
``knowledge/attention_families.yaml`` is mirrored here as
``_resolve_attention_family`` and applied to **bags of kernel names**
that match real DSV2/DSV3.2/DSV4/Qwen3/Mamba traces. The test fails if
the combination of categories emitted by ``common.categories_and_roles``
no longer resolves to the expected paper-aligned family.

This is the executable form of the "must_have / must_not_have"
signatures listed in attention_families.yaml. Family names follow the
DeepSeek papers (``mla`` / ``dsa`` / ``csa`` / ``hca`` / ``gqa`` / …),
NOT the CANN backend class name. DSA (V3.2) and CSA (V4) both route
through AscendSFABackend on Ascend, but they are different paper
architectures distinguished by whether a Compressor kernel is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ascend_profile import common  # noqa: E402


def _categories_from_kernels(names: list[str]) -> set[str]:
    cats: set[str] = set()
    for n in names:
        c, _ = common.categories_and_roles(n, "", "")
        cats.update(c)
    return cats


def _resolve_attention_family(names: list[str]) -> str:
    """Apply the cheat-sheet from attention_families.yaml.

    Returns one of: ``csa``, ``hca``, ``dsa``, ``mla``, ``linear``,
    ``gqa``, ``fa``, ``attn``. KVComp overlay is appended as ``+kvc``.
    """
    cats = _categories_from_kernels(names)

    has_compressor = "attention.kv_compressor" in cats
    has_indexer = "attention.lightning_indexer" in cats
    has_sparse_sharedkv = (
        "attention.sparse_sharedkv" in cats
        or "attention.sparse_sharedkv.metadata" in cats
    )
    has_dense_fia = "attention.gqa_or_mha" in cats
    has_mla_marker = (
        "attention.mla" in cats
        or "attention.mla.preprocess" in cats
        or "attention.mla.kv_norm_rope_cache" in cats
        or "attention.mla.v_up_proj" in cats
    )

    if has_compressor and has_indexer and has_sparse_sharedkv:
        base = "csa"
    elif has_compressor and has_dense_fia and not has_indexer and not has_sparse_sharedkv:
        base = "hca"
    elif has_indexer and has_sparse_sharedkv and not has_compressor:
        base = "dsa"
    elif has_mla_marker and not (has_compressor or has_indexer or has_sparse_sharedkv):
        base = "mla"
    elif "attention.linear_or_mamba" in cats:
        base = "linear"
    elif has_dense_fia:
        base = "gqa"
    else:
        base = "attn"

    if "attention.kvcomp.topk" in cats:
        return f"{base}+kvc"
    return base


# Real-trace kernel bags. Each list is a *subset* of the kernels that
# appear in one attention block for the named family; the full block is
# bigger (RoPE, norm, BMM, etc.) but the listed ones are the unique
# signature kernels.

_FIXTURES: list[tuple[str, list[str], str]] = [
    # ---------- DeepSeek V2 / V3 MLA decode ----------
    (
        "DSV2_V3_MLA_decode",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "FusedInferAttentionScoreV2",
            "TransposeQuantBatchMatmul",
            "InterleaveRope",
        ],
        "mla",
    ),
    (
        "DSV2_V3_MLA_prefill",
        [
            "KvRmsNormRopeCache",
            "FusedInferAttentionScore",
            "InterleaveRope",
        ],
        "mla",
    ),
    (
        "DSV2_V3_MLA_with_canonical_CANN_name",
        # CANN op_list canonical names. We accept all three spellings.
        [
            "MlaProlog",
            "KvRmsNormRopeCache",
            "FusedInferAttentionScore",
        ],
        "mla",
    ),
    # ---------- DeepSeek V3.2 = DSA (NOT csa; NOT mla) ----------
    # DSA = Lightning Indexer + Sparse-SharedKV, NO Compressor.
    # DSA is built on MLA (V3.2 paper §4), so MLAPO and KvRmsNormRopeCache
    # still appear, but the sparse signatures must win.
    (
        "DSV3.2_DSA_decode",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "InterleaveRope",
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "BatchMatmulTranspose",
        ],
        "dsa",
    ),
    (
        "DSV3.2_DSA_prefill",
        [
            "KvRmsNormRopeCache",
            "QuantLightningIndexer",
            "KVQuantSparseAttnSharedKV",
            "IndexerCompressEpilogV2",
            "InPlacePartialRotaryMul",
        ],
        "dsa",
    ),
    # ---------- DeepSeek V4 = CSA (main layers) ----------
    # CSA = KV Compressor + Lightning Indexer + Sparse-SharedKV. The
    # presence of the Compressor kernel is what distinguishes V4 CSA
    # from V3.2 DSA.
    (
        "DSV4_CSA_prefill",
        [
            "KVQuantSparseAttnSharedKV",
            "KVQuantSparseAttnSharedKVMetadata",
            "QuantLightningIndexer",
            "QuantLightningIndexerMetadata",
            "Compressor",
            "KVCompressEpilog",
            "IndexerCompressEpilogV2",
            "InPlacePartialRotaryMul",
        ],
        "csa",
    ),
    (
        "DSV4_CSA_decode_with_MLAPO_reuse",
        [
            "MlaPreprocess",
            "KvRmsNormRopeCache",
            "InterleaveRope",
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "Compressor",
            "BatchMatmulTranspose",
        ],
        "csa",
    ),
    # ---------- DeepSeek V4 = HCA (alternating layers, heuristic) ----------
    # HCA = Compressor + dense FIA, no indexer, no sparse-sharedkv.
    (
        "DSV4_HCA_heuristic",
        [
            "Compressor",
            "KVCompressEpilog",
            "FusedInferAttentionScore",
            "InterleaveRope",
        ],
        "hca",
    ),
    # ---------- Dense GQA (Llama / Qwen / Mistral) ----------
    (
        "Qwen3_dense_decode",
        [
            "FusedInferAttentionScore",
            "NpuRotaryEmbedding",
        ],
        "gqa",
    ),
    (
        "Llama_dense_prefill",
        [
            "UnpadFlashAttention",
            "NpuRotaryEmbedding",
        ],
        "gqa",
    ),
    # ---------- Linear / Mamba / GDN ----------
    (
        "Mamba2_attn_layer",
        ["CausalConv1d"],
        "linear",
    ),
    # ---------- KVComp overlays ----------
    (
        "DSV3.2_DSA_with_kvcomp",
        [
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "NpuHammingDistTopK",
            "NpuSignBitsPack",
        ],
        "dsa+kvc",
    ),
    (
        "DSV4_CSA_with_kvcomp",
        [
            "KVQuantSparseAttnSharedKV",
            "QuantLightningIndexer",
            "Compressor",
            "NpuHammingDistTopK",
        ],
        "csa+kvc",
    ),
    (
        "DSV2_MLA_with_kvcomp",
        [
            "KvRmsNormRopeCache",
            "FusedInferAttentionScoreV2",
            "NpuHammingDistTopK",
        ],
        "mla+kvc",
    ),
    (
        "Dense_with_kvcomp",
        [
            "FusedInferAttentionScore",
            "NpuHammingDistTopK",
        ],
        "gqa+kvc",
    ),
]


@pytest.mark.parametrize(
    "label,kernels,expected_family",
    _FIXTURES,
    ids=[c[0] for c in _FIXTURES],
)
def test_attention_family_resolution(label, kernels, expected_family):
    got = _resolve_attention_family(kernels)
    assert got == expected_family, (
        f"fixture {label}: kernels {kernels} resolved to family {got!r}, "
        f"expected {expected_family!r}"
    )


def test_csa_vs_dsa_distinguished_by_compressor():
    """The Compressor kernel is the *only* difference between a V3.2 DSA
    layer and a V4 CSA layer at the kernel level. Drop the Compressor
    from a CSA bag → it must reclassify as DSA. Add a Compressor back →
    must reclassify as CSA.
    """
    csa_bag = ["KVQuantSparseAttnSharedKV", "QuantLightningIndexer", "Compressor"]
    dsa_bag = ["KVQuantSparseAttnSharedKV", "QuantLightningIndexer"]

    assert _resolve_attention_family(csa_bag) == "csa"
    assert _resolve_attention_family(dsa_bag) == "dsa"


def test_mla_signature_disjoint_from_sparse():
    """A pure MLA bag (no Compressor, no Indexer, no Sparse-SharedKV)
    must resolve to ``mla``. A pure sparse bag must NOT pick up the
    ``mla`` family label even when it shares the MLA preprocess kernel.
    """
    mla_only = ["MlaPreprocess", "KvRmsNormRopeCache", "TransposeQuantBatchMatmul"]
    dsa_with_mla_reuse = [
        "MlaPreprocess",
        "KvRmsNormRopeCache",
        "KVQuantSparseAttnSharedKV",
        "QuantLightningIndexer",
    ]
    csa_with_mla_reuse = dsa_with_mla_reuse + ["Compressor"]

    assert _resolve_attention_family(mla_only) == "mla"
    assert _resolve_attention_family(dsa_with_mla_reuse) == "dsa"
    assert _resolve_attention_family(csa_with_mla_reuse) == "csa"


def test_block_head_hc_prefix_does_not_pollute_attention_family():
    """The HC* block-head prefix kernels appear before BOTH attention
    and MoE blocks. Adding them to a DSA bag must not flip the family,
    must not introduce moe.gating, must not pretend to be SFA-specific.
    """
    dsa_with_hc = [
        "HCPreSinkhorn",
        "HCPreInvRMS",
        "HCPost",
        "KVQuantSparseAttnSharedKV",
        "QuantLightningIndexer",
    ]
    assert _resolve_attention_family(dsa_with_hc) == "dsa"
    cats = _categories_from_kernels(dsa_with_hc)
    assert "block_head.mhc_prefix" in cats
    assert "moe.gating" not in cats
